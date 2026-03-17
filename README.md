# Genie Prototype – AI-Powered Document Intelligence Engine

## Overview

Genie Prototype is an **AI-powered backend service designed to analyze documents and measure semantic relationships between them using modern NLP and LLM techniques**.

The system uses **text embeddings and large language models (LLMs)** to understand document meaning and calculate similarity between uploaded documents. This enables intelligent document comparison, knowledge discovery, and semantic search.

The project demonstrates how to build a **modular AI backend system integrating LLM chains, embeddings, and scalable API architecture**.

---

## Key Features

• AI-powered document similarity analysis
• Semantic embeddings for text representation
• LLMChain integration for intelligent processing
• Modular backend architecture
• Environment-based configuration management
• Scalable service design for AI applications

---

## System Architecture

The project follows a modular backend architecture separating responsibilities into independent layers.

```
Client Request
      │
      ▼
API Routes
      │
      ▼
Business Logic Layer
      │
      ▼
Embedding Engine + LLM Chain
      │
      ▼
Similarity Processing
      │
      ▼
Response to Client
```

---

## Project Structure

```
Genie_prototype
│
├── const/
│   └── configuration constants
│
├── core/
│   └── core AI components and initialization
│
├── logic/
│   └── document processing and similarity logic
│
├── route/
│   └── API route definitions
│
├── main.py
│   └── application entry point
│
├── .env.example
│   └── environment configuration template
│
├── requirements.txt
│   └── project dependencies
│
└── README.md
```

---

## How It Works

1. **Document Input**

Users upload documents or provide textual input.

2. **Text Processing**

The system extracts and cleans document text.

3. **Embedding Generation**

Documents are converted into vector embeddings using NLP models.

4. **LLM Processing**

LLMChain processes document context to enhance semantic understanding.

5. **Similarity Computation**

Vector similarity algorithms calculate how closely documents are related.

6. **Results**

The system returns a similarity score representing how closely documents match.

---

## Technologies Used

* Python
* Natural Language Processing (NLP)
* LLMChain
* Embeddings
* AI Model Integration
* REST API Architecture

---

## Installation

Clone the repository

```
git clone https://github.com/Mahee1911/Genie_prototype.git
```

Navigate into the project directory

```
cd Genie_prototype
```

Install dependencies

```
pip install -r requirements.txt
```

Configure environment variables

```
cp .env.example .env
```

---

## Running the Application

Start the backend service

```
python main.py
```

The API service will start and begin accepting requests.

---

## Potential Use Cases

- Document similarity detection
- Knowledge discovery systems
- Enterprise document intelligence
- Research paper comparison
- AI-powered document search


## Author

Mahee Gadhiya

GitHub:
https://github.com/Mahee1911
