# RetriPDF 📄 | Serverless RAG Chatbot

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)](https://www.langchain.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**RetriPDF** is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDF documents in real-time. 

Built with **LangChain** and **Streamlit**, it features a serverless architecture deployable on Hugging Face Spaces, utilizing the **Zephyr-7B** model for high-quality, free inference without requiring local GPU resources.

---

## 🏗️ Architecture

The system follows a modular RAG pipeline:

1.  **Ingestion**: PDFs are loaded and split into chunks using `RecursiveCharacterTextSplitter`.
2.  **Embedding**: Text chunks are converted to vectors using `sentence-transformers/all-MiniLM-L6-v2`.
3.  **Vector Store**: Embeddings are stored in **ChromaDB** (ephemeral/persistent modes supported).
4.  **Retrieval**: Semantic search retrieves the top-k most relevant context chunks.
5.  **Generation**: The **Zephyr-7B-beta** LLM (via Hugging Face Inference API) generates accurate answers based *strictly* on the retrieved context.

---

## 🚀 Features

* **Hybrid Deployment**: Runs locally with Ollama or in the cloud with Hugging Face Inference API.
* **Session Memory**: Maintains conversation history for context-aware follow-up questions.
* **Ephemeral File Handling**: Optimized for stateless container environments (like Docker/Spaces) using `/tmp` directory management.
* **Professional UI**: Custom CSS styling with dark mode and glassmorphism design.
* **Security**: Implemented XSRF protection bypass for iframe embedding.

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Orchestration**: LangChain
* **Vector DB**: ChromaDB
* **LLM Engine**: Hugging Face Inference API (Zephyr-7B)
* **Embeddings**: HuggingFaceEmbeddings (all-MiniLM-L6-v2)

---

## ⚡ Installation & Local Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/RetriPDF.git](https://github.com/yourusername/RetriPDF.git)
    cd RetriPDF
    ```

---

## ☁️ Deployment (Hugging Face Spaces)

This project is configured for direct deployment to Hugging Face Spaces.

1.  Create a new **Streamlit Space**.
2.  Add your `HF_TOKEN` in the **Settings > Variables and secrets** tab.
3.  Ensure `packages.txt` (if needed) and `requirements.txt` are present.
4.  The `app.py` handles ephemeral file storage in `/tmp` automatically.

---

## 📸 Screenshots

*(Add a screenshot of your app here)*

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
