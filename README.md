# 📚 Context-Aware RAG System

A **Context-Aware Retrieval-Augmented Generation (RAG)** system that improves retrieval quality by comparing:

* 🔹 **Raw Vector Search**
* 🔹 **AI-Enhanced Query Expansion**

This project demonstrates how intelligent query rewriting improves semantic search results in a local RAG pipeline.

---

## 🚀 Features

* 📄 Document ingestion (PDF, TXT)
* ✂️ Intelligent chunking using text splitters
* 🔍 Embedding-based semantic search
* 🧠 Query expansion (mock LLM)
* ⚡ FAISS / Chroma-based vector storage
* 📊 Retrieval benchmarking (Strategy A vs Strategy B)
* 🧪 Modular & extensible architecture

---

## 🏗️ Project Structure

```
context_aware_RAG_system/
│
├── data/                  # Input documents
├── src/
│   ├── ingestion.py       # Data loading & chunking
│   ├── embedding.py       # Embedding generation
│   ├── vector_store.py    # Vector DB logic
│   ├── retriever.py       # Retrieval strategies
│   ├── query_expansion.py # Query rewriting (mock LLM)
│   └── benchmark.py       # Strategy comparison
│
├── tests/                 # Pytest test cases
├── config.py              # Configurations
├── requirements.txt
└── main.py                # Entry point
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sanjeevspuranik/context_aware_RAG_system.git
cd context_aware_RAG_system
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add Your Data

Place your documents inside the `data/` folder:

```
data/
 ├── file1.pdf
 ├── file2.txt
```
Place your queries inside the `queries.txt` file:

```
each line should have one query
```

---

## ▶️ Running the Project

### Run Full Pipeline

```bash
python main.py
```

---

### Run Benchmark (Core Task)

```bash
python src/benchmark.py
```

This will output:

* 🔹 Top retrieved chunks (Raw search)
* 🔹 Top retrieved chunks (Expanded query)
* 🔹 Comparison results (JSON / table)

---

## 🧠 Retrieval Strategies

### 🔹 Strategy A: Raw Vector Search

* Direct embedding of user query
* Standard similarity search

### 🔹 Strategy B: Query Expansion

* Query rewritten using mock LLM
* Improved semantic alignment
* Better retrieval relevance

---

## 📊 Example Query

```
"How does the system handle peak load?"
```

Output includes:

* Top 3 results (Strategy A)
* Top 3 results (Strategy B)
* Comparative analysis

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## ⚙️ Configuration

Modify `config.py`:

```python
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📈 Evaluation

The system benchmarks:

* Retrieval quality
* Semantic relevance
* Effectiveness of query expansion

This aligns with **context-aware retrieval systems**, where preprocessing queries significantly improves search results ([Elite AI Tools][1])

---

## 🔧 Tech Stack

* Python
* LangChain
* Sentence Transformers
* FAISS / ChromaDB
* PyMuPDF

---

## 🚧 Future Improvements

* Replace mock LLM with real LLM (OpenAI / Gemini)
* Add hybrid search (BM25 + vector)
* UI (Streamlit / Next.js)
* Multi-modal RAG (images + text)
* LangGraph agent integration

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork and improve the system.

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

Inspired by modern RAG architectures combining:

* Embeddings
* Vector search
* Query rewriting for better retrieval accuracy ([VRRaj][2])

---

[1]: https://eliteai.tools/agent-skills/rag-infrastructure?utm_source=chatgpt.com "rag-infrastructure - AI Agent skill"
[2]: https://vrraj.github.io/chat-with-rag/?utm_source=chatgpt.com "Chat with RAG Tool-Assisted Multi-Provider RAG Framework | Chat with RAG"
