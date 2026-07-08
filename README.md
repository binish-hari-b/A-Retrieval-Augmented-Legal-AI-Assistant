# OSIRIS: Retrieval Augmented Legal AI Assistant

OSIRIS is a fully local Legal AI Assistant built on a custom implementation of the Llama 3.2 inference architecture and a Retrieval Augmented Generation (RAG) pipeline for grounded question answering over Indian legal statutes.

Unlike conventional LLM applications that rely on high level inference libraries, OSIRIS implements the complete transformer inference pipeline from scratch in PyTorch and combines it with semantic retrieval over a curated corpus of Indian legislation. The system performs entirely local inference using pretrained Llama 3.2 1B Instruct weights and does not depend on external APIs or cloud hosted language models.

---

## Key Features

- Fully local inference with no external APIs
- Custom implementation of the Llama 3.2 inference architecture in PyTorch
- Retrieval Augmented Generation over Indian legal statutes
- Semantic retrieval using BGE embeddings and FAISS
- Grounded legal responses generated only from retrieved statutory context
- Offline legal question answering
- Modular Retrieval Augmented Generation pipeline

---

# Custom Llama 3.2 Inference Engine

The transformer inference engine was implemented entirely from scratch in PyTorch.

Implemented components include:

- Token Embeddings
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- Feed Forward Networks using SiLU activation
- Residual Connections
- Causal Self Attention
- Autoregressive Text Generation
- Inference using pretrained Llama 3.2 1B Instruct weights

The project does not use Hugging Face Transformer inference pipelines. The transformer architecture, attention mechanism, decoding pipeline and text generation were implemented manually.

---

# Retrieval Augmented Generation Pipeline

The legal retrieval system was developed from scratch and consists of:

- Legal document preprocessing
- Semantic text chunking
- BGE embedding generation
- FAISS vector indexing
- Cosine similarity based semantic retrieval
- Prompt construction
- Grounded response generation

```
                    User Query
                        │
                        ▼
          Sentence Transformer (BGE)
                        │
                        ▼
               FAISS Vector Search
                        │
                        ▼
          Top K Relevant Legal Chunks
                        │
                        ▼
             Prompt Construction
                        │
                        ▼
      Custom Llama 3.2 Inference Engine
                        │
                        ▼
            Grounded Legal Response
```

---

# Legal Knowledge Base

The current knowledge base includes major Indian legislation including:

- Constitution of India
- Bharatiya Nyaya Sanhita
- Bharatiya Nagarik Suraksha Sanhita
- Bharatiya Sakshya Adhiniyam
- Indian Contract Act
- Consumer Protection Act
- Information Technology Act
- Companies Act
- Right to Information Act
- Arbitration and Conciliation Act
- Hindu Marriage Act
- Motor Vehicles Act
- Negotiable Instruments Act
- Transfer of Property Act
- Sale of Goods Act
- Specific Relief Act
- Central Goods and Services Tax Act
- Protection of Women from Domestic Violence Act

The legal corpus is stored as structured text documents and indexed for semantic retrieval.

---

# Technologies Used

- Python
- PyTorch
- FAISS
- Sentence Transformers
- Hugging Face Tokenizers
- LangChain Text Splitters

---

# Repository Structure

```
OSIRIS
│
├── data
│   └── Indian legal statutes
│
├── rag
│   └── rag_pipeline.py
│
├── llama_model.ipynb
├── config.json
└── README.md
```

---

# Model Weights

The pretrained Llama 3.2 1B Instruct model weights are **not included** due to GitHub file size limitations.
Download the official model weights separately and place them inside the `models` directory before running the project.

---

# Running the Project

1. Download the official Llama 3.2 1B Instruct model weights.
2. Place the weights inside the `models` directory.
3. Ensure the legal text corpus is available inside the `data` directory.
4. Open `llama_model.ipynb`.
5. Run the notebook and begin querying the legal knowledge base.

---

# Example Queries

- What is coercion under the Indian Contract Act?
- Define consideration.
- What is a contingent contract?
- Differentiate coercion and undue influence.
- Who is competent to contract?
- Define primary evidence.
- Who bears the burden of proof?
- Define a consumer under the Consumer Protection Act.
- When can a police officer arrest without a warrant?

---

# Current Limitations

- Uses a lightweight 1B parameter language model
- Retrieval quality depends on semantic search and document chunking
- No reranking stage
- No KV cache optimization
- Responses are limited to the indexed legal corpus

---

# Future Improvements

- KV cache based decoding
- Adaptive retrieval
- Web based interface
- Larger legal knowledge base
- Citation highlighting

---

# Disclaimer

OSIRIS is an educational project demonstrating transformer inference and Retrieval Augmented Generation.

The generated responses should not be considered professional legal advice.
