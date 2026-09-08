# 🚘 AI-Powered Vehicle Inspection Knowledge Assistant (RAG)

## Project Screenshot

![alt text](Image.png)

## Overview

The **AI-Powered Vehicle Inspection Knowledge Assistant** is a Retrieval-Augmented Generation (RAG) application designed for the automotive vehicle inspection domain. It enables users to ask natural-language questions about vehicle inspection procedures, damage annotation guidelines, validation processes, and operational manuals.

The system combines **document processing, semantic embeddings, vector search, retrieval-augmented generation, conversational memory, and Large Language Models (LLMs)** to provide context-aware answers with source and page references.

The solution is implemented as an interactive **Streamlit application** and is designed to support faster and more consistent access to automotive inspection knowledge.

---

## Business Context

Automotive inspection operations depend on large volumes of inspection manuals, annotation guidelines, validation procedures, and operational documentation.

Manually searching these documents can be time-consuming and may result in inconsistent interpretation of inspection rules.

This project addresses the problem by providing an AI-powered knowledge assistant that can:

- Retrieve relevant information from inspection documentation.
- Answer domain-specific questions using retrieved context.
- Provide source and page references for traceability.
- Support conversational, multi-turn interactions.
- Reduce the time required to manually search large documentation.

The solution is particularly relevant to **vehicle inspection, damage annotation, validation, and automotive quality-control workflows**.

---

## Key Features

- **PDF Document Ingestion** – Upload and process vehicle inspection and operational documents.
- **Intelligent Text Chunking** – Splits documents into retrieval-friendly chunks with configurable chunk size and overlap.
- **Semantic Search** – Uses HuggingFace `multilingual-e5-large` embeddings to represent document content as vectors.
- **Vector Retrieval** – ChromaDB performs similarity-based retrieval of relevant document sections.
- **RAG-Based Question Answering** – Retrieved context is passed to an LLM to generate grounded responses.
- **Conversational Memory** – LangGraph maintains conversation state for multi-turn interactions.
- **Source Traceability** – Responses include the source document and relevant page numbers.
- **Interactive Streamlit UI** – Provides an easy-to-use interface for document upload and question answering.
- **Local Document Processing** – Source documents and vector data are maintained within the application environment.

## RAG Workflow

The application implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for automotive vehicle inspection and damage annotation knowledge retrieval.

### Workflow

1. **Document Ingestion**
   Upload automotive inspection manuals and operational documents in PDF format.

2. **Document Processing**
   Extract text from documents and divide the content into configurable chunks using chunk size and overlap parameters.

3. **Embedding Generation**
   Generate semantic embeddings using the `intfloat/multilingual-e5-large` embedding model.

4. **Vector Storage**
   Store document embeddings in **ChromaDB** for persistent vector-based retrieval.

5. **Query Processing**
   Convert natural-language user questions into semantic search queries.

6. **Similarity Search**
   Retrieve the most relevant document chunks from the vector database based on semantic similarity.

7. **Context Management**
   Pass the retrieved document context and user query through a **LangGraph** workflow.

8. **Response Generation**
   Generate context-grounded responses using the **Groq-hosted `openai/gpt-oss-20b` LLM**.

9. **Source Attribution**
   Display relevant source documents and page references to improve response traceability and verification.

---

## System Architecture

```text
                ┌──────────────────────┐
                │   PDF Documents      │
                │ Inspection Manuals    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Document Processing  │
                │ Text Extraction      │
                │ Chunking             │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Embedding Generation │
                │ multilingual-e5-large│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      ChromaDB        │
                │  Vector Storage      │
                └──────────┬───────────┘
                           │
                 User Query
                           │
                           ▼
                ┌──────────────────────┐
                │ Similarity Retrieval │
                │ Relevant Chunks      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      LangGraph       │
                │ Context Management   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Groq LLM           │
                │ openai/gpt-oss-20b   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Context-Grounded     │
                │ Response + Sources   │
                └──────────────────────┘
```

---

## Tech Stack

| Category               | Technology                             |
| ---------------------- | -------------------------------------- |
| Programming Language   | Python                                 |
| RAG Framework          | LangChain                              |
| Workflow Orchestration | LangGraph                              |
| User Interface         | Streamlit                              |
| Embedding Model        | `intfloat/multilingual-e5-large`       |
| Vector Database        | ChromaDB                               |
| Large Language Model   | Groq — `openai/gpt-oss-20b`            |
| Machine Learning       | PyTorch                                |
| NLP / Embeddings       | HuggingFace                            |
| Document Processing    | PDF Parsing, Text Extraction, Chunking |
| Configuration          | Python Virtual Environment, `.env`     |

---

## Project Highlights

* Developed an **end-to-end RAG-based knowledge assistant** for automotive vehicle inspection and damage annotation documentation.
* Implemented **semantic document retrieval** using HuggingFace embeddings and ChromaDB.
* Designed a configurable document-processing pipeline with adjustable **chunk size and chunk overlap**.
* Integrated **LangGraph** to manage retrieval and conversational workflow state.
* Integrated a **Groq-hosted LLM** for context-grounded response generation.
* Implemented **source and page-level attribution** to improve response verification and traceability.
* Developed an interactive **Streamlit application** for document upload, querying, and conversational interaction.
* Implemented persistent vector storage to support repeated querying without reprocessing documents.
* Applied practical concepts from **Machine Learning, NLP, Generative AI, Information Retrieval, and LLM application development**.
* Troubleshot and resolved practical issues involving **dependencies, embeddings, vector dimensions, model integration, and application configuration**.

---

## Key RAG Components

### 1. Document Ingestion

PDF documents are loaded and processed to extract textual content required for downstream retrieval.

### 2. Text Chunking

Large documents are divided into smaller overlapping chunks to improve retrieval granularity and preserve contextual relationships between sections.

### 3. Semantic Embeddings

The `intfloat/multilingual-e5-large` model converts document chunks and user queries into numerical vector representations.

### 4. Vector Retrieval

ChromaDB stores the generated embeddings and performs similarity-based retrieval to identify relevant document sections.

### 5. Context-Augmented Generation

Retrieved document content is provided as context to the LLM, enabling responses to be generated based on the available inspection documentation.

### 6. Source Attribution

Retrieved document metadata is preserved so that users can identify the source document and page associated with the generated response.

---

## Business Benefits

### Faster Knowledge Retrieval

Enables inspection teams to quickly locate relevant procedures, annotation guidelines, and operational information from large documentation sets.

### Improved Consistency

Provides responses grounded in documented inspection guidelines, supporting standardized interpretation of vehicle inspection information.

### Reduced Manual Search

Reduces the time required to manually search through lengthy inspection manuals and operational documents.

### Improved Traceability

Source and page references allow users to verify the documentation used to generate a response.

### Scalable Architecture

The architecture can be extended to additional automotive knowledge sources such as service manuals, repair procedures, technical specifications, and compliance documentation.

---

## Future Enhancements

* Extend the system to **Multimodal RAG** for images, diagrams, tables, and visual inspection documentation.
* Integrate the **YOLOv8 Vehicle Damage Detection System** with the RAG knowledge assistant.
* Implement **Hybrid Search** combining semantic and keyword-based retrieval.
* Add a **reranking stage** to improve retrieved-context relevance.
* Evaluate retrieval performance using **Precision@K, Recall@K, and Mean Reciprocal Rank (MRR)**.
* Add **voice-based interaction** using speech-to-text and text-to-speech.
* Implement automated **vehicle inspection report generation** using retrieved evidence.
* Develop multi-step **agentic workflows with LangGraph** for complex inspection tasks.
* Add automated evaluation for **retrieval quality and LLM response grounding**.

---

## Conclusion

The **AI-Powered Vehicle Inspection Knowledge Assistant** demonstrates the application of **Machine Learning, Natural Language Processing, Generative AI, and Retrieval-Augmented Generation** to an automotive knowledge-retrieval problem.

The system combines **Python, LangChain, LangGraph, HuggingFace Embeddings, ChromaDB, Groq LLM, PyTorch, and Streamlit** to provide a conversational interface for querying vehicle inspection documentation.

This project demonstrates practical ML Engineer skills in:

* Document processing
* Semantic embeddings
* Vector databases
* Information retrieval
* RAG architecture
* LLM integration
* Conversational workflows
* NLP
* Application development
* Model and dependency debugging
* End-to-end GenAI pipeline development

