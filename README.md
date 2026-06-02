# AI RAG PDF Chatbot

A professional AI portfolio project that allows users to upload a PDF and ask questions about the document using a Retrieval-Augmented Generation style workflow.

## Live Demo

Add Streamlit deployment link here after deployment.

## Project Overview

This application demonstrates a practical document AI workflow:

1. Upload a PDF
2. Extract readable text
3. Split the text into searchable chunks
4. Retrieve the most relevant sections based on a user question
5. Generate grounded responses from the document content

This version is designed to work without paid API keys, which makes it easy to deploy and test on Streamlit Community Cloud. It can later be upgraded with OpenAI, LangChain, ChromaDB, FAISS, or other embedding/vector database tools.

## Why This Project Matters

RAG systems are one of the most important skills in modern AI development. Companies use this type of system to analyze:

- contracts
- business reports
- financial documents
- research papers
- internal policies
- client documents
- knowledge bases

This project shows practical understanding of document intelligence, retrieval systems, and enterprise AI workflows.

## Features

- PDF upload
- PDF text extraction
- document chunking
- local vector search
- RAG-style question answering
- chat interface
- relevance scoring
- document preview
- manual source search
- Streamlit web interface
- deployment-ready structure

## Tech Stack

- Python
- Streamlit
- PyPDF
- Scikit-learn
- TF-IDF retrieval
- Cosine similarity

## Project Structure

```text
ai-rag-pdf-chatbot/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── sample_documents/
```

## How To Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/ensarmaxhuni/ai-rag-pdf-chatbot.git
cd ai-rag-pdf-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use these settings:

```text
Repository: ensarmaxhuni/ai-rag-pdf-chatbot
Branch: main
Main file path: app.py
Python version: 3.12
Secrets: none required
```

## Future Improvements

- Add OpenAI-generated final answers
- Add embeddings
- Add ChromaDB or FAISS vector storage
- Add multiple PDF upload support
- Add page-level citations
- Add chat export
- Add document comparison
- Add enterprise authentication
- Add production API backend

## Portfolio Value

This project demonstrates:

- RAG architecture
- document AI
- retrieval systems
- Python development
- Streamlit deployment
- AI product thinking
- business automation potential

## Author

Ensar Maxhuni

- Portfolio: https://ensarmaxhuni.github.io
- GitHub: https://github.com/ensarmaxhuni
