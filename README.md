# AI-Powered Vehicle Inspection Knowledge Assistant (RAG)
![alt text](image.png)
## Overview

This project is a Retrieval-Augmented Generation (RAG) based AI assistant designed for the automotive vehicle inspection domain. Inspired by ClearQuote's vehicle inspection workflow, the system enables users to interact with vehicle inspection manuals, damage annotation guidelines, and operational documents using natural language queries.

The solution combines semantic search, vector databases, and Large Language Models (LLMs) to deliver accurate, context-aware, and source-backed answers in real time.

## Business Context

Vehicle inspection platforms are widely used by rental and commercial fleet operators to ensure:

* Guided vehicle image capture
* Automated damage detection
* Alerts for newly identified damage
* Transparent vehicle handovers
* Increased driver accountability
* Reduced damage-related operational costs

Inspection teams often rely on extensive documentation and annotation manuals. Manually searching these documents is time-consuming and can lead to inconsistencies.

This project addresses that challenge by providing an AI-powered assistant capable of retrieving relevant information instantly from vehicle inspection and damage annotation documents.

---

## Key Features

* PDF document ingestion and processing
* Intelligent text chunking for efficient retrieval
* Semantic search using HuggingFace embeddings
* Context-aware document retrieval with ChromaDB
* Natural language question answering using Llama 3.1-8B
* Conversational memory for multi-turn interactions
* Interactive Streamlit web application
* Source-backed responses for improved reliability
* Privacy-first architecture with local document processing

---

## System Architecture

1. Upload vehicle inspection manuals and annotation documents.
2. Documents are split into semantic chunks.
3. HuggingFace embeddings convert text into vector representations.
4. ChromaDB stores and retrieves relevant document sections.
5. Retrieved context is provided to ChatGroq's Llama 3.1-8B model.
6. The LLM generates accurate and context-aware responses.
7. Results are displayed through an interactive Streamlit interface.

---

## Tech Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| Framework            | LangChain                      |
| User Interface       | Streamlit                      |
| Embedding Model      | intfloat/multilingual-e5-large |
| Vector Database      | ChromaDB                       |
| Large Language Model | ChatGroq (Llama 3.1-8B)        |
| Document Processing  | PDF Parsing & Text Chunking    |

---

## Project Highlights

* Developed an end-to-end RAG pipeline for automotive inspection documentation.
* Implemented semantic retrieval using HuggingFace multilingual embeddings and ChromaDB.
* Built a conversational AI assistant capable of answering vehicle inspection and damage annotation queries with source-aware responses.
* Deployed a user-friendly Streamlit application for real-time document interaction.
* Designed a privacy-focused architecture where documents remain within the local environment.

---

## Benefits

### Faster Knowledge Access

Retrieve relevant inspection procedures and annotation guidelines within seconds.

### Improved Consistency

Provides standardized responses based on approved documentation.

### Reduced Manual Effort

Eliminates the need to manually search through large PDF manuals.

### Scalable Architecture

Can be extended to support service manuals, repair procedures, compliance documents, and technical knowledge bases.

### Automotive Industry Relevance

Supports AI-driven vehicle inspection workflows and damage assessment operations.

---

## Future Enhancements

* Multimodal RAG for document images and diagrams
* Integration with vehicle damage detection models (YOLOv8)
* Voice-based inspection assistant
* Automated inspection report generation
* Agentic workflows using LangGraph
* Multi-document knowledge graph retrieval

---

## Conclusion

The AI-Powered Vehicle Inspection Knowledge Assistant demonstrates the practical application of Generative AI and Retrieval-Augmented Generation in the automotive industry.

By combining HuggingFace embeddings, ChromaDB, LangChain, Streamlit, and ChatGroq's Llama 3.1-8B model, the solution delivers accurate, context-aware, and source-backed answers from vehicle inspection documentation, helping inspection teams improve efficiency, consistency, and decision-making.
