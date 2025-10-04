[English](./README.md) | [한국어](./README.ko.md) | [日本語](./README.ja.md)

---

# 🧠 JLPT Question Generation Learning Helper - AI Service

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](#-tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](#-tech-stack)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-purple.svg)](#-tech-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This is the core AI service for the "Generative AI-based JLPT Question Generation Learning Helper". Built on FastAPI, it interacts with various Large Language Models (LLMs) using LangChain. It provides more accurate and context-aware JLPT question generation and chatbot responses through a Retrieval-Augmented Generation (RAG) architecture.

## ✨ Features

- **🤖 Dynamic Question Generation**: Generates JLPT questions (vocabulary, grammar, reading) in real-time using LLMs.
- **💬 RAG-based Chatbot**: Provides accurate, evidence-based answers to user questions by leveraging a knowledge base stored in a FAISS vector store.
- **🔄 Multi-LLM Support**: Flexibly switch between and use various language models like OpenAI, Google Gemini, and Groq as needed.
- **⚡️ High-Performance Async API**: Ensures high throughput and fast response times with FastAPI.

## 🏛️ Architecture: RAG (Retrieval-Augmented Generation)

This service adopts the RAG architecture to complement the limitations of LLMs.

1.  **Input**: A user submits a request to generate a question or asks a question.
2.  **Retrieve**: The system searches a `FAISS` vector store for documents most relevant to the input.
3.  **Augment**: The retrieved documents (Context) and the original input are combined into a new, augmented prompt.
4.  **Generate**: This augmented prompt is sent via `LangChain` to an LLM (e.g., GPT-4, Gemini) to generate a contextually accurate answer or question.

This approach reduces hallucinations and produces highly specialized results for a specific domain (JLPT).

## 🛠️ Tech Stack

| Category | Technology / Library | Description |
| :--- | :--- | :--- |
| **Language** | Python | 3.12 |
| **Web Framework** | FastAPI, Uvicorn | Asynchronous API server |
| **AI Framework** | LangChain | LLM application development |
| **LLM Integration**| OpenAI, Google GenAI, Groq | |
| **Vector Search** | FAISS (faiss-cpu) | Embedding vector search for RAG |
| **Environment** | python-dotenv | |
| **Data Handling** | Pydantic, unstructured | |

## 📂 Project Structure

```
app/
    main.py                   # FastAPI application entry point and router setup
    chatbot/                  # Logic for the general chatbot
    problem_generator/        # Logic for JLPT question generation
    user_question_chatbot/    # Logic for the RAG chatbot that answers user questions
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.12 or higher
- pip

### 2. Installation

In the project root directory, run the following command to install dependencies.
```bash
pip install -r requirements.txt
```

### 3. Environment Variable Setup

Create a `.env` file in the project root and enter the API keys for the LLMs you intend to use.

```
# .env

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key_here"

# Google GenAI API Key
GOOGLE_API_KEY="your_google_api_key_here"

# Groq API Key
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Running the Development Server

Run the command below to start the Uvicorn development server. The `--reload` flag ensures the server restarts automatically on code changes.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 API Docs & Endpoints

FastAPI automatically generates API documentation compliant with the OpenAPI 3.0 specification. After running the dev server, open your web browser and navigate to **`http://localhost:8000/docs`** to view and test all APIs via the Swagger UI.

- `POST /generate-problem`: Requests the generation of a new JLPT problem.
- `POST /chat`: Asks a question to the RAG-based chatbot.

## 🐳 Running with Docker

1.  **Build the Docker image**
    ```bash
    docker build -t jlpt-ai-service:latest .
    ```

2.  **Run the Docker container**
    Run the container by injecting the API keys from your `.env` file as environment variables.
    ```bash
    docker run -p 8000:8000 \
      -e OPENAI_API_KEY="your_openai_api_key" \
      -e GOOGLE_API_KEY="your_google_api_key" \
      -e GROQ_API_KEY="your_groq_api_key" \
      jlpt-ai-service:latest
    ```

## 🤝 Contributing

Contributions are always welcome! Please create an issue or submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
