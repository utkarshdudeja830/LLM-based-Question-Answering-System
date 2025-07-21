# LLM-based-Question-Answering-System

A LLM-based question-answering system leveraging Retrieval-Augmented Generation (RAG) to provide concise and accurate answers about NASA and its missions, including the Voyager program. The system combines document retrieval with a large language model to deliver contextually relevant responses.

## Features
- Loads and processes text documents from a specified file.
- Uses HuggingFace embeddings and FAISS for efficient document retrieval.
- Integrates a language model (Mistral-7B) for natural language answers.
- Interactive console interface for querying NASA-related information.
- Configurable via environment variables.

## Overview
The system employs a **Retrieval-Augmented Generation (RAG)** approach:
- **Document Processing**: Text documents are split into chunks (default: 750 characters, 150-character overlap) using `RecursiveCharacterTextSplitter` to preserve context.
- **Embedding and Vector Store**: Document chunks are converted into embeddings using `all-MiniLM-L6-v2` and stored in a FAISS vector store for fast similarity-based retrieval.
- **Language Model**: The `unsloth/mistral-7b-instruct-v0.1-bnb-4bit` model generates responses, optimized with quantization and parameters like `top_p=0.95`, `top_k=40`, and `temperature=0.2`.
- **RetrievalQA Chain**: Retrieves top `k` (default: 5) relevant chunks to provide context for the language model, using a custom prompt to ensure concise, context-grounded answers.

## Project Structure
```
.
├── README.md                # Project documentation
├── config
│   └── variables.env        # Environment variables (e.g., DOCUMENT_PATH)
├── data
│   └── nasa_info.txt        # Input text file with NASA-related information
├── requirements.txt         # Python dependencies
└── src
    └── agent.py             # Main script for the QA system
```

## Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt` (e.g., `langchain`, `transformers`, `faiss-cpu`, `python-dotenv`).
- A `.env` file in `config` with:
  ```
  DOCUMENT_PATH=data/nasa_info.txt
  ```

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LLM-based-Question-Answering-System
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure `data/nasa_info.txt` contains NASA-related content.
5. Create `config/variables.env` with the path to the data file.

## Usage
1. Run the QA system:
   ```bash
   python src/agent.py
   ```
2. Enter questions about NASA or its missions in the console.
3. Type `exit` or `quit` to stop.

## Example
```bash
$ python src/agent.py
Welcome to the NASA QA System! Type 'exit' or 'quit' to stop.
Enter your question: What is the Voyager program?
Answer:
The Voyager program consists of two spacecraft, Voyager 1 and Voyager 2, launched by NASA in 1977 to explore the outer planets and eventually study interstellar space. Both spacecraft visited Jupiter and Saturn, with Voyager 2 also exploring Uranus and Neptune. They continue to send data back to Earth, providing insights into the far reaches of our solar system.
```

## Notes
- The Mistral-7B model requires significant computational resources (GPU recommended).
- Ensure `nasa_info.txt` is populated with accurate content for meaningful responses.
- FAISS enables efficient retrieval, but performance depends on input document quality.
