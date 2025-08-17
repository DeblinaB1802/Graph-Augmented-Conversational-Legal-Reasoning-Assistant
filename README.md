# Legal Document Analyzer
## Graph-Augmented Conversational Legal Assistant

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-0.1+-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A sophisticated legal document analysis tool that combines Retrieval-Augmented Generation (RAG) with dynamic knowledge graphs to provide intelligent Q&A capabilities over legal documents. Built with LangChain, OpenAI GPT-3.5, and NetworkX for enhanced semantic understanding.

## 🚀 Features

### Core Capabilities
- **📄 PDF Document Processing**: Upload and analyze legal documents in PDF format
- **🤖 Intelligent Q&A**: Ask natural language questions about legal content
- **🔍 Multi-Query Expansion**: Enhanced retrieval through diverse query perspectives
- **⚖️ Legal Entity Recognition**: Automatic extraction of sections, acts, cases, concepts, procedures, and penalties
- **📊 Knowledge Graph Visualization**: Interactive graph representation of document entities and relationships

### Advanced Features
- **🔗 Hybrid Query Processing**: Dual-mode operation for content-based and semantic queries
- **📈 Reciprocal Rank Fusion (RRF)**: Advanced document ranking and retrieval
- **💾 ChromaDB Vector Storage**: Efficient semantic search capabilities
- **📊 Document Analytics**: Comprehensive statistics and entity analysis
- **💬 Chat History**: Persistent conversation tracking
- **📤 Export Functionality**: Download analysis results in JSON/CSV formats

## 🏗️ Architecture

### Hybrid RAG Pipeline
```
PDF Upload → Document Processing → Dual Storage:
                                   ├── Vector Store (ChromaDB)
                                   └── Knowledge Graph (NetworkX)
                                            ↓
Query Classification → Route to:
                      ├── Graph Traversal (Entity/Content queries)
                      └── Vector Search + LLM (Semantic queries)
```

### Key Components
1. **Document Processor**: PDF parsing and text chunking
2. **Entity Extractor**: LLM-powered legal entity recognition
3. **Vector Store**: ChromaDB with OpenAI embeddings
4. **Knowledge Graph**: NetworkX-based entity relationship mapping
5. **Query Router**: Intelligent query classification system
6. **Response Generator**: Context-aware answer generation

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- At least 4GB RAM (recommended for large documents)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/legal-document-analyzer.git
   cd legal-document-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

## 📦 Dependencies

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.1.0
chromadb>=0.4.0
networkx>=3.0
pandas>=1.5.0
matplotlib>=3.5.0
pdfplumber>=0.9.0
python-dotenv>=1.0.0
openai>=1.0.0
```

## 🚀 Usage

### Starting the Application

1. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8501`
   - Enter your OpenAI API key in the sidebar

### Using the Assistant

#### 1. Document Upload
- Upload a legal PDF document using the file uploader
- Wait for processing (document chunking, entity extraction, graph building)

#### 2. Query Types

**Graph Knowledge Base Queries** (routed automatically):
```
"What sections are mentioned in the document?"
"Which acts or laws are referenced?"
"Give me a summary of the document content"
"What legal concepts are discussed?"
```

**Semantic RAG Queries** (routed automatically):
```
"What is the punishment for theft under IPC?"
"Explain the concept of bail in criminal law"
"What are the grounds for arrest without warrant?"
```

#### 3. Features Overview

- **📊 Analytics Dashboard**: View entity distribution and frequency analysis
- **🔍 Retrieved Documents**: Inspect source documents for answers
- **💬 Chat History**: Track conversation context
- **📈 Graph Visualization**: Explore entity relationships
- **📤 Export Options**: Download analysis results

## 🔧 Configuration

### Customizing Entity Extraction

Modify the `extract_legal_entity` method in `LegalDocumentGraph` class to adjust entity categories:

```python
entity_extraction_prompt = """Extract legal entities from the following text. Return a JSON object with these categories:
- sections: Legal sections (e.g., "Section 302 IPC")
- acts: Acts/laws (e.g., "Indian Penal Code")
- cases: Case names or precedents
- concepts: Legal concepts (e.g., "bail", "jurisdiction")
- procedures: Legal procedures
- penalties: Punishments or penalties
"""
```

### Adjusting Chunk Size

Modify text splitting parameters in `process_uploaded_file`:

```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,  # Adjust based on document complexity
    chunk_overlap=200  # Overlap for context preservation
)
```

## 📊 Query Routing Logic

The system automatically determines query routing based on keywords:

### Graph Queries (Content Analysis)
- Document summaries and overviews
- Entity listings (sections, acts, cases)
- Content structure analysis

### RAG Queries (Semantic Search)
- Legal interpretations and explanations
- Specific legal advice
- Case law applications

## 🎯 Performance Optimization

### For Large Documents
1. **Increase chunk size** for faster processing
2. **Reduce entity extraction complexity** for speed
3. **Limit graph visualization** for performance

### For Better Accuracy
1. **Decrease chunk size** for granular retrieval
2. **Increase overlap** for context preservation
3. **Use multiple query perspectives** for robustness

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to initialize LLM"**
   - Verify OpenAI API key is correct
   - Check internet connectivity
   - Ensure sufficient API credits

2. **"No content found in PDF"**
   - Verify PDF is text-based (not scanned images)
   - Try a different PDF file
   - Check file permissions

3. **Slow processing**
   - Large documents take longer to process
   - Consider reducing chunk size
   - Ensure adequate system memory

### Debug Mode

Enable detailed logging by modifying the logging configuration:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Analytics and Insights

### Document Statistics
- Total entities extracted
- Entity type distribution
- Most frequent entities
- Document chunk analysis

### Performance Metrics
- Response time tracking
- Query success rates
- Entity extraction accuracy

## 🔒 Security Considerations

- API keys are not stored permanently
- Temporary files are cleaned up after processing
- No data is sent to external services except OpenAI API
- Local processing ensures document privacy

## 🛣️ Roadmap

### Version 2.0 Features
- [ ] Support for multiple document formats (DOCX, TXT)
- [ ] Advanced graph algorithms for entity relationship discovery
- [ ] Integration with legal databases
- [ ] Multi-language support
- [ ] Cloud deployment options

### Enhancements
- [ ] Improved entity extraction accuracy
- [ ] Custom legal domain fine-tuning
- [ ] Batch document processing
- [ ] API endpoint creation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- **LangChain** for the RAG pipeline framework
- **OpenAI** for the GPT-3.5 language model
- **ChromaDB** for vector storage capabilities
- **NetworkX** for graph processing
- **Streamlit** for the web interface



---

**Built with ❤️ for the legal technology community**
