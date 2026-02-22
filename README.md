# 🔍 DataCo Global Policies RAG Pipeline

<p align="center">
  <a href="https://colab.research.google.com/github/opalclouds/DataCo-global-policies-RAG-pipeline/blob/main/DataCo_policies_RAG.ipynb" target="_parent">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-orange" />
  <img src="https://img.shields.io/badge/NetworkX-Graph%20RAG-green" />
  <img src="https://img.shields.io/badge/SentenceTransformers-MiniLM-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

<p align="center">
  A <strong>Graph-enhanced Retrieval-Augmented Generation (RAG)</strong> pipeline for intelligently querying DataCo Global's corporate policy documents — combining semantic vector search with knowledge graph traversal for richer, more contextually aware retrieval.
</p>

---

## 🧠 What Makes This Special

Most RAG pipelines stop at vector search. This one goes further.

By layering a **NetworkX knowledge graph** on top of FAISS semantic search, the pipeline can traverse *semantically related* document chunks — not just the top-K nearest neighbors — giving you more complete, connected answers from across policy documents.

```
PDF Documents → Text Extraction → Chunking → Embeddings
                                                   ↓
                                      ┌─────────────────────┐
                                      │   FAISS Index        │  ← Fast vector search
                                      │   NetworkX Graph     │  ← Semantic relationships
                                      └─────────────────────┘
                                                   ↓
                                          Query → Top-K Chunks
                                                + Graph Neighbors
                                                = Richer Context
```

---

## ✨ Features

- **📄 PDF Ingestion** — Bulk extract text from multiple policy PDFs using `pdfplumber`
- **✂️ Smart Chunking** — Sliding window chunking with configurable size and overlap to preserve context at boundaries
- **🧬 Semantic Embeddings** — Powered by `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality sentence embeddings
- **⚡ FAISS Vector Index** — L2 similarity search across thousands of chunks in milliseconds
- **🕸️ Knowledge Graph (GraphRAG)** — NetworkX graph connects semantically similar chunks (cosine similarity > 0.75), enabling neighborhood expansion during retrieval
- **💾 Persistent Storage** — Embeddings saved as `.parquet`, FAISS index persisted as `.index` for reuse
- **☁️ Google Colab Ready** — Kaggle API integration for seamless dataset download in Colab

---

## 🗂️ Project Structure

```
DataCo-global-policies-RAG-pipeline/
│
├── DataCo_policies_RAG.ipynb   # Main notebook (run in Colab)
├── embeddings.parquet           # Persisted chunk embeddings
├── embedding.index              # FAISS index file
└── README.md
```

---

## 🚀 Getting Started

### 1. Open in Colab

Click the badge at the top or go directly to the notebook. All dependencies install automatically.

### 2. Set Up Kaggle API

Upload your `kaggle.json` credentials when prompted. The notebook will:
- Download the [DataCo Global Policy Dataset](https://www.kaggle.com/datasets/sghhim/dataco-global-policy-dataset)
- Unzip and load all PDFs automatically

### 3. Run the Pipeline

Execute cells in order to:
1. Extract and chunk policy text
2. Generate and store embeddings
3. Build the FAISS index and knowledge graph
4. Query the system interactively

---

## 🔬 How It Works

### Stage 1 — Document Ingestion
`pdfplumber` reads every `.pdf` in the working directory and extracts raw text page by page.

### Stage 2 — Chunking
Text is split into 800-character chunks with 100-character overlap, ensuring no sentence or clause is lost at a boundary. Each chunk is assigned a unique ID (`filename_chunk_N`).

### Stage 3 — Embedding
`all-MiniLM-L6-v2` encodes every chunk into a 384-dimensional dense vector, balancing speed and semantic quality.

### Stage 4 — FAISS Index
All embeddings are loaded into a `faiss.IndexFlatL2` for fast approximate nearest-neighbor search.

### Stage 5 — Knowledge Graph Construction
A NetworkX graph is built where:
- **Nodes** = document chunks
- **Edges** = cosine similarity > 0.75 between any two chunks

This creates a semantic map of the entire policy corpus.

### Stage 6 — Graph-Enhanced Query
When a query is submitted:
1. It is embedded using the same model
2. Top-5 most similar chunks are retrieved via FAISS
3. Their **graph neighbors** are added to expand context
4. The final retrieved set is richer and more topically complete than vector search alone

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pdfplumber` | PDF text extraction |
| `sentence-transformers` | Semantic embeddings (`all-MiniLM-L6-v2`) |
| `FAISS` | High-speed vector similarity search |
| `NetworkX` | Knowledge graph for GraphRAG |
| `scikit-learn` | Cosine similarity matrix |
| `pandas` | Data management |
| `NumPy` | Numerical operations |
| `Kaggle API` | Dataset download |

---

## 📊 Dataset

**DataCo Global Policy Dataset** — a collection of corporate policy PDFs covering governance, compliance, and operational guidelines.

- Source: [Kaggle](https://www.kaggle.com/datasets/sghhim/dataco-global-policy-dataset)
- Format: Multiple PDF files
- Access: Requires a free Kaggle account

---

## 💡 Use Cases

- **HR & Compliance Teams** — Instantly query any policy without reading dozens of documents
- **Legal Review** — Cross-reference related policy clauses automatically
- **Onboarding** — Let new employees ask natural language questions about company policies
- **Audit Preparation** — Retrieve policy context across connected documents

---

## 🔮 Roadmap

- [ ] Add an LLM generation layer (e.g., GPT-4 / Gemini) for full Q&A responses
- [ ] Build a Gradio or Streamlit UI
- [ ] Support hybrid BM25 + semantic search
- [ ] Add metadata filtering (by document, date, category)
- [ ] Visualize the knowledge graph interactively

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Built with ❤️ for smarter document intelligence</p>
