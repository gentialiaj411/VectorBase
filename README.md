# MiniVector 🚀
### Local Semantic Search & AI Research Assistant

MiniVector is a high-performance, privacy-focused semantic search engine designed to help researchers navigate complex academic papers. Unlike traditional keyword search, MiniVector uses **vector embeddings** to understand the *meaning* behind your query, allowing you to find relevant papers even if they don't use the exact same words.

## 💡 Why I Built This
I built MiniVector to explore the intersection of **Information Retrieval** and **Generative AI**. I wanted to solve a real problem: finding and understanding specific insights within dense technical literature without relying on expensive cloud APIs or sacrificing data privacy.

This project demonstrates my ability to:
- Build **Full-Stack Applications** (React + FastAPI).
- Implement **RAG (Retrieval-Augmented Generation)** pipelines from scratch.
- Work with **Vector Databases** and high-dimensional data.
- Integrate **Local LLMs** (Ollama/Llama 3.2) for offline inference.

## ✨ Key Features
- **Semantic Search**: Finds papers based on conceptual similarity using high-dimensional vector embeddings.
- **Citation Topology**: Interactive graph visualization to explore how papers reference each other.
- **AI Research Assistant**: Chat with your papers! A local LLM (Llama 3.2) answers questions based on the paper's content.
- **100% Local**: Runs entirely on your machine—no data leaves your system.

## 🛠️ How to Run

### Prerequisites
- Python 3.9+
- Node.js & npm
- [Ollama](https://ollama.com/) (for AI chat)

### 1. Setup Backend
```bash
# Clone the repo
git clone https://github.com/yourusername/minivector.git
cd minivector

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python api/server.py
```

### 2. Setup Frontend
```bash
cd frontend
npm install
npm start
```

### 3. Enable AI (Optional)
To use the chat feature, make sure Ollama is running with the Llama 3.2 model:
```bash
ollama pull llama3.2
ollama serve
```

Visit `http://localhost:3000` to start searching!

## 🚀 Future Roadmap
Here are some exciting features I plan to implement to push this project further:

- **GraphRAG**: Combining the citation graph with vector search to allow the AI to "reason" across multiple papers.
- **Live arXiv Ingestion**: A pipeline to automatically fetch and index the latest papers daily.
- **Multi-Modal Search**: Enabling search by figures, diagrams, and equations.
- **Distributed Indexing**: Scaling the vector engine to handle millions of documents using sharding.

---
*Built with ❤️ by [Your Name]*
