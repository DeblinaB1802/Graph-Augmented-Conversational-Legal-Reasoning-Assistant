import os
import re
import time
import streamlit as st
import logging
import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import chromadb
import pdfplumber
from dotenv import load_dotenv
from collections import defaultdict
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)
st.header("📚 Legal Document Analyzer")
st.subheader("Upload a legal document and get intelligent analysis with Q&A capabilities")

# API Key Input
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:", 
    type="password",
    help="Your API key is not stored and only used for this session"
)
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()
os.environ['OPENAI_API_KEY'] = api_key

# Initialize LLM with error handling
@st.cache_resource
def initialize_llm():
    try:
        return ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

llm = initialize_llm()
if not llm:
    st.warning("⚠️ Please ensure that LLM is successfully initialized.")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_graph" not in st.session_state:
    st.session_state.document_graph = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None

# Graph Knowledge Base Class
class LegalDocumentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relationships = {}
        self.document_metadata = {}

    def extract_legal_entity(self, text, llm):
        """Extract legal entities from the documents using LLM."""
        entity_extraction_prompt = """Extract legal entities from the following text. Return a JSON object with these categories:
        - sections: List of legal sections mentioned (e.g., "Section 302 IPC", "Article 21")
        - acts: List of acts/laws mentioned (e.g., "Indian Penal Code", "Constitution of India")
        - cases: List of case names or legal precedents
        - concepts: List of legal concepts (e.g., "bail", "jurisdiction", "criminal liability")
        - procedures: List of legal procedures mentioned
        - penalties: List of punishments or penalties mentioned

        Text: {text}

        Return only valid JSON:
        """

        try:
            response = llm.invoke(entity_extraction_prompt.format(text=text))
            content = response.content if hasattr(response, 'content') else str(response)
            try:
                entities = json.loads(content)
            except:
                clean_content = re.sub(r'\s+', ' ', content)
                patterns = {
                    "sections": r'\b(?:Section|Sec\.?)\s+\d+[A-Z]?(?:\s+of\s+(?:the\s+)?(?:IPC|CrPC|CPC|Evidence Act))?\b',
                    "acts": r'\b(?:Indian Penal Code|Constitution of India|Criminal Procedure Code|Civil Procedure Code|Evidence Act)\b',
                    "cases": r'\b(?:[A-Z][a-z]+\s+v(?:s\.?|ersus)\s+[A-Z][a-z]+)\b',
                    "concepts": r'\b(?:bail|jurisdiction|liability|mens rea|actus reus|due process|negligence|criminal liability)\b',
                    "procedures": r'\b(?:FIR|filing of FIR|arrest|remand|charge sheet|trial|appeal)\b',
                    "penalties": r'\b(?:imprisonment(?:\s+up\s+to)?\s+\d+\s+(?:years?|months?)|fine|capital punishment|death penalty|life imprisonment)\b'
                }
                entities = {key: re.findall(pat, clean_content, flags=re.IGNORECASE) for key, pat in patterns.items()}
            return entities

        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return {
                "sections": [],
                "acts": [],
                "cases": [],
                "concepts": [],
                "procedures": [],
                "penalties": []
            }

    def build_graph_from_documents(self, documents, llm, filename):
        """Build knowledge graph from document chunks."""
        self.document_metadata = {
            "filename": filename or "uploaded_document",
            "total_chunks": len(documents),
            "creation_time": time.time()
        }

        all_entities = defaultdict(set)
        chunk_entities = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, doc in enumerate(documents):
            status_text.text(f"Processing chunk {i+1}/{len(documents)}")
            progress_bar.progress((i+1)/len(documents))

            entities = self.extract_legal_entity(doc.page_content, llm)
            chunk_entities.append(entities)
            chunk_id = f"chunk_{i}"
            self.graph.add_node(chunk_id, type="chunk", content=doc.page_content)

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity and entity.strip():
                        entity_id = f"{entity_type}_{entity}"
                        self.graph.add_node(entity_id, type=entity_type, name=entity)
                        self.graph.add_edge(chunk_id, entity_id, relationship='contains')
                        all_entities[entity_type].add(entity)

        for entity_type, entity_set in all_entities.items():
            for entity in entity_set:
                entity_id = f"{entity_type}_{entity}"
                self.entities[entity_id] = {
                    "type": entity_type,
                    "name": entity,
                    "frequency": sum(1 for chunk in chunk_entities if entity in chunk.get(entity_type, []))
                }

        progress_bar.empty()
        status_text.empty()

        st.success(f"✅ Graph built successfully with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

    def get_document_summary(self):
        """Get summary statistics of the document."""
        if not self.graph.nodes:
            return {}

        summary = {
            "total_entities": len([node for node in self.graph.nodes if not node.startswith("chunk_")]),
            "total_chunks": len([node for node in self.graph.nodes if node.startswith("chunk_")]),
            "entity_breakdown": {},
            "most_frequent_entities": {}
        }

        for entity_id, entity_info in self.entities.items():
            entity_type = entity_info['type']
            if entity_type not in summary["entity_breakdown"]:
                summary['entity_breakdown'][entity_type] = 0
            summary['entity_breakdown'][entity_type] += 1

        for entity_type in summary['entity_breakdown'].keys():
            type_entities = [(entity_id, entity_info) for entity_id, entity_info in self.entities.items() if entity_info['type'] == entity_type]
            if type_entities:
                most_frequent = max(type_entities, key=lambda x: x[1]['frequency'])
                summary['most_frequent_entities'][entity_type] = {
                    "name": most_frequent[1]['name'],
                    "frequency": most_frequent[1]['frequency']
                }
        return summary

    def _identify_relevant_entity_types(self, query: str) -> list:
        """Identify which entity types are relevant to the query."""
        query_lower = query.lower()
        relevant_types = []

        type_keywords = {
            'sections': ['section', 'sections', 'sec', 'provision', 'provisions', 'article', 'articles'],
            'acts': ['act', 'acts', 'law', 'laws', 'statute', 'statutes', 'code', 'codes'],
            'cases': ['case', 'cases', 'precedent', 'precedents', 'judgment', 'judgments', 'ruling', 'rulings'],
            'concepts': ['concept', 'concepts', 'principle', 'principles', 'doctrine', 'doctrines', 'theory', 'theories'],
            'procedures': ['procedure', 'procedures', 'process', 'processes', 'step', 'steps', 'method', 'methods'],
            'penalties': ['penalty', 'penalties', 'punishment', 'punishments', 'fine', 'fines', 'imprisonment', 'sentence']
        }

        for entity_type, keywords in type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_types.append(entity_type)

        if not relevant_types:
            relevant_types = list(type_keywords.keys())
        return relevant_types

    def _collect_entities_by_type(self, entity_types: list) -> dict:
        """Collect all entities for the specified types."""
        collected_entities = {}

        for entity_type in entity_types:
            entities_of_type = []
            for entity_id, entity_info in self.entities.items():
                if entity_info['type'] == entity_type:
                    entities_of_type.append({
                        'name': entity_info['name'],
                        'frequency': entity_info['frequency']
                    })

            # Sort by frequency (most frequent first)
            entities_of_type.sort(key=lambda x: x['frequency'], reverse=True)
            collected_entities[entity_type] = entities_of_type
        return collected_entities

    def _get_relevant_chunks(self, collected_entities: dict, max_chunks: int = 5) -> list:
        """Get document chunks that are connected to relevant entities."""
        relevant_chunks = set()
        relevant_entity_names = set()
        for entities in collected_entities.values():
            for entity in entities:
                relevant_entity_names.add(entity['name'].lower())

        for node_id in self.graph.nodes:
            if node_id.startswith("chunk_"):
                for neighbor in self.graph.successors(node_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if 'name' in neighbor_data:
                        entity_name = neighbor_data['name'].lower()
                        if entity_name in relevant_entity_names:
                            chunk_content = self.graph.nodes[node_id].get('content', '')
                            relevant_chunks.add(chunk_content)
                            break

        relevant_chunks = list(relevant_chunks)
        return relevant_chunks[:max_chunks]


    def _build_llm_context(self, collected_entities: dict, relevant_chunks: list) -> str:
        """Build context string for LLM from collected entities and chunks."""
        context = ""

        for entity_type, entities in collected_entities.items():
            if entities:
                context += f"\n**{entity_type.title()}:**\n"
                for entity in entities:
                    context += f"- {entity['name']} (mentioned {entity['frequency']} times)\n"

        if relevant_chunks:
            context += f"\n**Relevant Document Chunks:**\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context += f"{i}. {chunk}...\n"
        print(context)
        return context

    def _handle_dynamic_graph_query(self, query: str, llm) -> str:
        """Handle all graph-related queries dynamically using LLM."""
        if not self.entities:
            return "No entities have been extracted from the document yet. Please upload and process a document first."

        relevant_types = self._identify_relevant_entity_types(query)
        collected_entities = self._collect_entities_by_type(relevant_types)
        relevant_chunks = self._get_relevant_chunks(collected_entities)
        context = self._build_llm_context(collected_entities, relevant_chunks)

        llm_prompt = f"""
        You are a legal document analysis assistant. Based on the extracted entities and document information provided below, answer the user's question comprehensively and accurately.

        **Document Information:**
        - File: {self.document_metadata.get('filename', 'Unknown')}
        - Total document chunks: {self.document_metadata.get('total_chunks', 0)}

        **Extracted Entities:**
        {context}

        **User Question:** {query}

        Please provide a detailed and helpful answer based on the available information. If the question asks for specific entities (like sections, acts, cases, etc.),
        list them clearly with their frequencies. If it's a more analytical question, provide insights based on the entities and their relationships.

        Format your response clearly with appropriate headings and bullet points where helpful.
        """

        try:
            response = llm.invoke(llm_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error processing query with LLM: {str(e)}"

    def _handle_content_query(self) -> str:
        """Handle queries about document content."""
        if not self.document_metadata:
            return "No document has been uploaded yet."

        summary = self.get_document_summary()

        response = f"**Document Content Summary:**\n\n"
        response += f"📄 **File:** {self.document_metadata['filename']}\n"
        response += f"📊 **Total Chunks:** {summary.get('total_chunks', 0)}\n"
        response += f"🔍 **Total Entities:** {summary.get('total_entities', 0)}\n\n"

        if summary.get('entity_breakdown'):
            response += "____Entity Breakdown:**\n"
            for entity_type, count in summary['entity_breakdown'].items():
                response += f"- {entity_type.title()}: {count}\n"
            response += "\n"

        if summary.get('most_frequent_entities'):
            response += "**Most Frequent Entities:**\n"
            for entity_type, info in summary['most_frequent_entities'].items():
                response += f"- {entity_type.title()}: {info['name']} (mentioned {info['frequency']} times)\n"

        return response

    def query_graph(self, query: str, llm) -> str:
        """Query the knowledge graph based on natural language."""
        query_lower = query.lower()

        if any(phrase in query_lower for phrase in ["content of", "what is in", "summary of", "about the document", "overview of", "explain the document", "what does the file contain"]):
            return self._handle_content_query()

        return self._handle_dynamic_graph_query(query, llm)

# Prompt Templates
STRUCTURED_QUERY_TEMPLATE = """You are a legal expert assistant helping to improve search queries for better information retrieval 
from legal documents such as the Constitution, IPC, CrPC, Evidence Act, case law databases, and legal commentaries. 

Use the chat history only when the current user question is a direct follow-up, clarification, or elaboration request 
(e.g., questions that begin with phrases like 'explain more', 'elaborate', 'what about this', 'why is that', etc.).
Ignore the chat history entirely when the user introduces a new topic or question unrelated to previous conversation.

Given a user's original question, rewrite it into a more precise and well-structured version that includes:
- Relevant legal domain (criminal, constitutional, civil, corporate, etc.)
- Legal terminology and accurate phrasing
- Specific acts, sections, or legal doctrines if implied
- Clear and unambiguous language that reflects a legal research perspective

Example:
Original: "Can police arrest without warrant?"
Restructured: "Under what circumstances can the police make an arrest without a warrant as per Section 41 of the CrPC in criminal law?"

Original Question: {question}
Chat History Context: {chat_history}

Restructured Query:"""

MULTI_QUERY_TEMPLATE = """You are a legal assistant that generates multiple diverse versions of a user's question 
to improve document retrieval from Indian legal texts like IPC, CrPC, Evidence Act, Constitution, and court judgments.

Generate 5 alternative phrasings of the original question. Each variation should:
- Reflect different legal perspectives or doctrines
- Include relevant Act/Code references (e.g., Section 302 of IPC, Article 21 of the Constitution)
- Use precise legal language and terminology
- Address related aspects of the topic such as procedural law, case law relevance, or constitutional provisions

Original Question: {question}

Generate 5 diverse reformulations (one per line):"""

ANSWER_TEMPLATE = """You are a legal expert assistant. Use the provided context from legal statutes, case laws, and authoritative commentaries 
to answer the question with legal precision.

Instructions:
- Use only the information from the provided context
- Cite relevant sections, articles, or case names where applicable
- If the context does not provide a complete answer, say: "The provided context does not contain sufficient information to answer this question."
- Ensure the response is legally sound, concise, and aligned with Indian law

Context: {context}

Question: {question}

Answer:"""

# Create Prompt Templates
prompt_structured = ChatPromptTemplate.from_template(STRUCTURED_QUERY_TEMPLATE)
prompt_perspectives = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
prompt_answer = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

# Structured Query Creation Pipeline
generate_structured_query = (
    prompt_structured
    | llm
    | StrOutputParser()
)

# Multi Query Creation Pipeline
generate_multi_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [query.strip() for query in x.split("\n") if query.strip()])
)

def reciprocal_rank_fusion(results: list[list], top_k=5, k=30):
    """Implement Reciprocal Rank Fusion for combining multiple ranked lists."""
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    reranked_docs = [
        loads(doc_str) 
        for doc_str, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_docs[:top_k]

def process_uploaded_file(file):
    """Process uploaded PDF file and create temporary vectorstore + graph."""
    if not file:
        return None, None
        
    try:
        temp_path = r"c:\Users\debli\OneDrive\Desktop\CivilAssistant\temp_uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        
        with st.spinner("📄 Loading PDF document..."):
            loader = PDFPlumberLoader(temp_path)
            docs = loader.load()

        if not docs:
            st.error("No content found in the PDF file.")
            return None, None
        
        with st.spinner("✂️ Splitting document into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

        with st.spinner("🔗 Building knowledge graph..."):
            graph = LegalDocumentGraph()
            graph.build_graph_from_documents(splits, llm, file.name)

        with st.spinner("🔍 Creating vector database..."):
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                embedding_ctx_length=3000
            )
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        os.remove(temp_path)
        return vectorstore, graph
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None

def is_graph_query(question):
    """Determine if a question should be answered using the graph knowledge base."""
    graph_keywords = [
        # General document inquiries
        "content of", "what is in", "summary of", "about the document", "overview of", "explain the document",
        "uploaded file", "uploaded document", "document contains", "what does the file contain",
        
        # Section-based queries
        "sections mentioned", "which sections", "legal sections", "all sections", "sections covered", "identify sections",
        
        # Act/law-based queries
        "acts mentioned", "which acts", "laws mentioned", "legal acts", "identify laws", "acts referenced", "statutes mentioned",
        
        # Case law and precedent
        "cases mentioned", "case law", "precedents", "legal cases", "referenced cases", "important cases", "case references",
        
        # Legal concepts and doctrines
        "concepts", "legal concepts", "topics covered", "doctrines", "legal topics", "discussed concepts", "terminologies",
        
        # Procedures and processes
        "procedures", "legal procedures", "steps involved", "legal processes",
        
        # Penalties and punishments
        "penalties", "punishments", "types of punishment", "punishment mentioned", "fine", "imprisonment",
        
        # Entity lookup
        "mention", "included in the document", "referenced", "appears in the document", "related to the document",
    ]
    return any(keyword in question.lower() for keyword in graph_keywords)

# Main UI
st.markdown("### 📁 Upload Legal Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type=['pdf'],
    help="Upload a legal document (PDF format) to analyze"
)

# Process Uploaded File
if uploaded_file and (not hasattr(st.session_state, 'current_file') or st.session_state.current_file != uploaded_file.name):
    temp_vectorstore, temp_graph = process_uploaded_file(uploaded_file)
    if temp_vectorstore and temp_graph:
        st.session_state.vectorstore = temp_vectorstore
        st.session_state.document_graph = temp_graph
        st.session_state.current_file = uploaded_file.name
        st.session_state.uploaded_file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size
        }
        st.session_state.chat_history = []

# Main query interface
if st.session_state.vectorstore and st.session_state.document_graph:
    st.markdown("### 💬 Ask Questions About Your Document")
    
    question = st.text_input(
        "Enter your question:", 
        placeholder="e.g., What are the main legal provisions mentioned? OR What sections are covered in this document?",
        key="question_input"
    )

    # Main QA Processing
    if question:
        try:
            # Check if this is a graph query
            if is_graph_query(question):
                st.info("🔗 Using Graph Knowledge Base")
                with st.spinner("Querying knowledge graph..."):
                    answer = st.session_state.document_graph.query_graph(question, llm)
                    
                    st.write("### Answer:")
                    st.write(answer)
                    
                    # Update chat history
                    st.session_state.chat_history.append((question, answer))

            else:
                st.info("🔍 Searching document content")
            
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})

                # Prepare chat history context
                chat_history_str = "\n".join([
                    f"User: {que}\nAssistant: {ans}" 
                    for que, ans in st.session_state.chat_history[-3:]
                ])

                with st.spinner("Processing your question..."):
                    # Generate structured query
                    structured_query = generate_structured_query.invoke({
                        "question": question,
                        "chat_history": chat_history_str
                    })

                    # Generate multiple query perspectives
                    multi_queries = generate_multi_queries.invoke({
                        "question": structured_query,
                    })
                    
                    if not multi_queries:
                        multi_queries = [structured_query]

                    # Retrieve documents for each query
                    retrieved_docs = []
                    for query in multi_queries:
                        try:
                            docs = retriever.invoke(query)
                            retrieved_docs.append(docs)
                        except Exception as e:
                            logger.warning(f"Failed to retrieve for query '{query}': {str(e)}")
                    
                    if not retrieved_docs:
                        st.error("Failed to retrieve relevant documents.")
                        
                    
                    # Apply reciprocal rank fusion
                    fused_docs = reciprocal_rank_fusion(retrieved_docs)
                    
                    # Generate answer
                    start_time = time.time()
                    response = llm.invoke(prompt_answer.format(context=fused_docs, question=structured_query))
                    response_time = time.time() - start_time
                    
                    # Extract content from response
                    if hasattr(response, 'content'):
                        answer = response.content
                    else:
                        answer = str(response)
                    
                    # Display answer
                    st.write("### Answer:")
                    st.write(answer)

                    # Update chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Performance metrics
                    st.caption(f"Response time: {response_time:.2f} seconds")

                # Show retrieved documents
                with st.expander("📄 Retrieved Documents", expanded=False):
                    for i, doc in enumerate(fused_docs, 1):
                        st.write(f"**Document {i}:**")
                        st.write(doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Source: {doc.metadata}")
                        st.write("---")
                
                # Show generated queries for debugging
                with st.expander("🔍 Generated Queries", expanded=False):
                    st.write("**Structured Query:**")
                    st.write(structured_query)
                    st.write("**Multi-perspective Queries:**")
                    for i, query in enumerate(multi_queries, 1):
                        st.write(f"{i}. {query}")

        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
else:
    st.info("👆 Please upload a PDF document to start analyzing")

# Sidebar for chat history and document info
with st.sidebar:
    st.header("📋 Document Information")
    
    if st.session_state.uploaded_file_info:
        st.write(f"**📄 File:** {st.session_state.uploaded_file_info['name']}")
        st.write(f"**📏 Size:** {st.session_state.uploaded_file_info['size']:,} bytes")
        
        if st.session_state.document_graph:
            summary = st.session_state.document_graph.get_document_summary()
            st.write(f"**🔍 Entities:** {summary.get('total_entities', 0)}")
            st.write(f"**📚 Chunks:** {summary.get('total_chunks', 0)}")
            
            if summary.get('entity_breakdown'):
                st.write("**📊 Entity Types:**")
                for entity_type, count in summary['entity_breakdown'].items():
                    st.write(f"• {entity_type.title()}: {count}")
        
        if st.button("🗑️ Clear Document"):
            for key in ['vectorstore', 'document_graph', 'uploaded_file_info', 'current_file', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.write("No document uploaded yet")
    
    st.write("---")
    
    # Chat History
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {q[:30]}..."):
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {a[:100]}{'...' if len(a) > 100 else ''}")
        
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Help section
with st.expander("ℹ️ How to use Graph Knowledge Base", expanded=False):
    st.markdown("""
    **Graph Knowledge Base Features:**
    
    The enhanced Law Assistant now includes a graph knowledge base that can answer meta-questions about your uploaded documents:
    
    **Document Content Queries:**
    - "What is the content of the uploaded file?"
    - "Give me a summary of the document"
    - "What topics are covered in this document?"
    
    **Entity-Specific Queries:**
    - "Which legal sections are mentioned in the document?"
    - "What acts or laws are referenced?"
    - "Are there any case laws mentioned?"
    - "What legal concepts are discussed?"
    
    **How it works:**
    1. Upload a PDF document
    2. The system builds a knowledge graph extracting legal entities
    3. Ask content-based questions to get structured insights
    4. Use traditional queries for specific legal research
    
    **Query Types:**
    - 🔗 **Graph queries**: About document content, entities, structure
    - 🔍 **RAG queries**: Specific legal questions requiring detailed answers
    """)

# Graph visualization section
if st.session_state.document_graph and hasattr(st.session_state.document_graph, 'graph'):
    with st.expander("📊 Document Graph Visualization", expanded=False):
        if st.button("Generate Graph Visualization"):
            try:
                matplotlib.use('Agg')  # Use non-interactive backend
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get graph
                G = st.session_state.document_graph.graph
                
                # Create layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Draw nodes by type
                chunk_nodes = [n for n in G.nodes() if n.startswith('chunk_')]
                entity_nodes = [n for n in G.nodes() if not n.startswith('chunk_')]
                
                # Draw chunk nodes
                nx.draw_networkx_nodes(G, pos, nodelist=chunk_nodes, 
                                     node_color='lightblue', node_size=300, 
                                     alpha=0.7, ax=ax)
                
                # Draw entity nodes
                nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, 
                                     node_color='lightcoral', node_size=500, 
                                     alpha=0.8, ax=ax)
                
                # Draw edges
                nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
                
                # Add labels for entity nodes only (to avoid clutter)
                entity_labels = {n: G.nodes[n].get('name', n)[:15] + '...' if len(G.nodes[n].get('name', n)) > 15 else G.nodes[n].get('name', n) 
                               for n in entity_nodes}
                nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=8, ax=ax)
                
                ax.set_title("Document Knowledge Graph\n(Blue: Document Chunks, Red: Legal Entities)", 
                           fontsize=14, fontweight='bold')
                ax.axis('off')
                
                st.pyplot(fig)
                
                # Graph statistics
                st.write("**Graph Statistics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", len(G.nodes()))
                with col2:
                    st.metric("Total Edges", len(G.edges()))
                with col3:
                    st.metric("Connected Components", nx.number_connected_components(G.to_undirected()))
                
            except ImportError:
                st.warning("Matplotlib not available for graph visualization. Install matplotlib to see graph plots.")
            except Exception as e:
                st.error(f"Error generating graph visualization: {str(e)}")

# Advanced Analytics Section
if st.session_state.document_graph:
    with st.expander("📈 Document Analytics", expanded=False):
        if st.button("Generate Analytics Report"):
            try:
                # Get document summary
                summary = st.session_state.document_graph.get_document_summary()
                
                # Create analytics
                st.subheader("📊 Entity Distribution")
                
                if summary.get('entity_breakdown'):
                    # Create a simple bar chart using Streamlit
                    entity_df = pd.DataFrame([
                        {"Entity Type": k.title(), "Count": v} 
                        for k, v in summary['entity_breakdown'].items()
                    ])
                    st.bar_chart(entity_df.set_index("Entity Type"))
                
                # Most frequent entities table
                if summary.get('most_frequent_entities'):
                    st.subheader("🔝 Most Frequent Entities")
                    frequent_df = pd.DataFrame([
                        {
                            "Type": k.title(), 
                            "Entity": v['name'], 
                            "Frequency": v['frequency']
                        }
                        for k, v in summary['most_frequent_entities'].items()
                    ])
                    st.dataframe(frequent_df, use_container_width=True)
                
                # Entity details
                st.subheader("📋 All Entities")
                if st.session_state.document_graph.entities:
                    all_entities_df = pd.DataFrame([
                        {
                            "Type": info['type'].title(),
                            "Name": info['name'],
                            "Frequency": info['frequency']
                        }
                        for info in st.session_state.document_graph.entities.values()
                    ]).sort_values(["Type", "Frequency"], ascending=[True, False])
                    
                    st.dataframe(all_entities_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating analytics: {str(e)}")

# Export functionality
if st.session_state.document_graph:
    with st.expander("💾 Export Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Graph Data"):
                try:
                    # Prepare export data
                    export_data = {
                        "document_metadata": st.session_state.document_graph.document_metadata,
                        "entities": st.session_state.document_graph.entities,
                        "summary": st.session_state.document_graph.get_document_summary()
                    }
                    
                    # Convert to JSON
                    json_data = json.dumps(export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="Download Graph Data (JSON)",
                        data=json_data,
                        file_name=f"graph_data_{st.session_state.uploaded_file_info['name']}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error preparing export: {str(e)}")
        
        with col2:
            if st.button("Export Entity List"):
                try:
                    if st.session_state.document_graph.entities:
                        # Create CSV data
                        entities_data = []
                        for entity_id, info in st.session_state.document_graph.entities.items():
                            entities_data.append({
                                "ID": entity_id,
                                "Type": info['type'],
                                "Name": info['name'],
                                "Frequency": info['frequency']
                            })
                        
                        entities_df = pd.DataFrame(entities_data)
                        csv_data = entities_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Entities (CSV)",
                            data=csv_data,
                            file_name=f"entities_{st.session_state.uploaded_file_info['name']}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error preparing CSV export: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Enhanced Law Assistant with Graph Knowledge Base | 
    Supports both traditional RAG queries and document content analysis
</div>
""", unsafe_allow_html=True)