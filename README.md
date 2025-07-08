# 🧠 Knowva: AI-Powered Document Insight Assistant

Knowva is an intelligent, LLM-driven assistant that enables users to semantically query and extract insights from documents with ease. Built for business and healthcare applications, Knowva bridges the gap between raw data and decision-making by transforming complex reports into actionable answers using advanced AI/ML techniques.

---

## 🚀 Demo Preview
📽️ [Watch Demo Video](https://drive.google.com/file/d/18_d8DUBGf8V0bSe02btB3zAwUr5QjyAv/view?usp=sharing)  
*A walkthrough of Knowva's capabilities, UI, and real-time AI-powered document querying*

---

## 🔍 Features

- 🧠 **LLM-powered Q&A over documents** (PDF, DOCX, PPT, XML, TXT)
- 📊 **BI and KPI extraction** from unstructured data
- 🔎 **Semantic search** using **FAISS** vector database and **LangChain RAG**
- 🗂️ **Chunking & Prompt Engineering** for accurate retrieval
- 🧾 **Chat history** and **PDF export** of generated responses
- 🗣️ **Voice input support** for hands-free interaction
- 🌗 **Light/Dark Mode UI** via modern Streamlit design
- 🧪 **Medical & Business use-case** integration
- 📩 Future-ready **email-based reporting**

---

## 💡 AI/ML & LLM Capabilities

| Technique | Description |
|----------|-------------|
| **LangChain + RAG Pipelines** | Enables retrieval-augmented generation for context-aware question answering. |
| **FAISS Vector Store** | Powers semantic search with embedded document chunks. |
| **NVIDIA NIM + OpenAI APIs** | Dual backend support for scalable, fast, and accurate LLM inference. |
| **Prompt Engineering** | Crafted prompts for structured KPI extraction and summarization. |
| **Document Chunking & Embedding** | Preprocessing documents to optimize context retrieval and memory usage. |

---

## ⚙️ Architecture Overview

- **Frontend**: Streamlit + Custom Components
- **Backend**: FastAPI, LangChain, Docker
- **Inference**: OpenAI GPT models, NVIDIA NIM (Triton Inference Server)
- **Storage**: FAISS (vector DB), AWS S3 (optional)
- **Deployment**: Dockerized backend hosted on AWS EC2 (GPU-enabled)

---

## 📦 Tech Stack

- **Languages**: Python, Shell
- **Libraries**: LangChain, FAISS, OpenAI SDK, NVIDIA Triton/NIM, PyPDF2, python-docx, pytesseract
- **Frameworks**: FastAPI, Streamlit
- **Infra**: Docker, AWS EC2, CUDA GPU (for acceleration)

📌 Use Cases
📑 Automated Business Intelligence Reporting

🧬 Simplifying complex medical reports

📚 AI-assisted knowledge base querying

📊 Extracting KPIs from contracts, surveys, and unstructured PDFs

🧠 Future Enhancements
✅ OAuth login and user profile management

✅ Multi-document ingestion and memory-augmented context

✅ Native mobile support via React Native or Flutter

✅ Real-time collaboration & shared Q&A dashboards
