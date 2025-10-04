[English](./README.md) | [한국어](./README.ko.md) | [日本語](./README.ja.md)

---

# 🧠 JLPT 문제 생성 학습 도우미 - AI 서비스

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](#-tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](#-tech-stack)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-purple.svg)](#-tech-stack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

"생성형 AI를 통한 JLPT 문제 생성 학습 도우미"의 핵심 두뇌 역할을 하는 AI 서비스입니다. FastAPI를 기반으로 구축되었으며, LangChain을 활용하여 다양한 LLM(거대 언어 모델)과 상호작용합니다. RAG(검색 증강 생성) 아키텍처를 통해 보다 정확하고 맥락에 맞는 JLPT 문제 생성 및 챗봇 응답 기능을 제공합니다.

## ✨ 주요 기능

- **🤖 동적 문제 생성**: LLM을 활용하여 JLPT 시험 유형(어휘, 문법, 독해)에 맞는 문제를 실시간으로 생성.
- **💬 RAG 기반 챗봇**: FAISS 벡터 스토어에 저장된 지식 베이스를 활용하여, 사용자의 질문에 대해 정확하고 근거 있는 답변을 제공.
- **🔄 다중 LLM 지원**: OpenAI, Google Gemini, Groq 등 필요에 따라 다양한 언어 모델을 유연하게 교체하며 사용 가능.
- **⚡️ 고성능 비동기 API**: FastAPI를 통해 높은 처리량과 빠른 응답 속도를 보장.

## 🏛️ 아키텍처: RAG (검색 증강 생성)

이 서비스는 RAG(Retrieval-Augmented Generation) 아키텍처를 채택하여 LLM의 한계를 보완합니다.

1.  **질문/요청 (Input)**: 사용자가 문제 생성 요청 또는 질문을 입력합니다.
2.  **검색 (Retrieve)**: 입력된 내용과 가장 관련성이 높은 문서를 `FAISS` 벡터 스토어에서 검색합니다.
3.  **증강 (Augment)**: 검색된 문서(Context)와 원본 질문을 프롬프트에 함께 넣어 LLM에 전달할 준비를 합니다.
4.  **생성 (Generate)**: 증강된 프롬프트를 `LangChain`을 통해 LLM(예: GPT-4, Gemini)에 전달하여, 맥락에 맞는 정확한 답변 또는 문제를 생성합니다.

이 방식을 통해 환각(Hallucination) 현상을 줄이고, 특정 도메인(JLPT)에 대한 전문성 높은 결과를 얻을 수 있습니다.

## 🛠️ 기술 스택

| 구분 | 기술 / 라이브러리 | 설명 |
| :--- | :--- | :--- |
| **언어** | Python | 3.12 |
| **웹 프레임워크** | FastAPI, Uvicorn | 비동기 API 서버 |
| **AI 프레임워크** | LangChain | LLM 애플리케이션 개발 |
| **LLM 연동** | OpenAI, Google GenAI, Groq | |
| **벡터 검색** | FAISS (faiss-cpu) | RAG를 위한 임베딩 벡터 검색 |
| **환경변수 관리**| python-dotenv | |
| **데이터 처리** | Pydantic, unstructured | |

## 📂 프로젝트 구조

```
app/
├── main.py                   # FastAPI 애플리케이션 진입점 및 라우터 설정
├── chatbot/                  # 일반 챗봇 관련 로직
├── problem_generator/        # JLPT 문제 생성 로직
└── user_question_chatbot/    # 사용자 질문에 답변하는 RAG 챗봇 로직
```

## 🚀 시작하기

### 1. 사전 요구사항

- Python 3.12 이상
- pip

### 2. 설치

프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 의존성을 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고, 사용할 LLM의 API 키를 입력합니다.

```
# .env

# OpenAI API Key
OPENAI_API_KEY="your_openai_api_key_here"

# Google GenAI API Key
GOOGLE_API_KEY="your_google_api_key_here"

# Groq API Key
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. 개발 서버 실행

아래 명령어를 실행하면 Uvicorn 개발 서버가 시작됩니다. `--reload` 옵션 덕분에 코드 변경 시 서버가 자동으로 재시작됩니다.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 API 문서 및 엔드포인트

FastAPI는 OpenAPI 3.0 규격에 따라 API 문서를 자동으로 생성해 줍니다. 개발 서버 실행 후, 웹 브라우저에서 **`http://localhost:8000/docs`** 로 접속하면 Swagger UI를 통해 모든 API를 확인하고 직접 테스트해볼 수 있습니다.

- `POST /generate-problem`: 새로운 JLPT 문제 생성을 요청합니다.
- `POST /chat`: RAG 기반 챗봇에게 질문합니다.

## 🐳 Docker로 실행하기

1.  **Docker 이미지 빌드**
    ```bash
    docker build -t jlpt-ai-service:latest .
    ```

2.  **Docker 컨테이너 실행**
    `.env` 파일의 API 키들을 환경 변수로 주입하여 컨테이너를 실행합니다.
    ```bash
    docker run -p 8000:8000 \
      -e OPENAI_API_KEY="your_openai_api_key" \
      -e GOOGLE_API_KEY="your_google_api_key" \
      -e GROQ_API_KEY="your_groq_api_key" \
      jlpt-ai-service:latest
    ```

## 🤝 기여하기

기여는 언제나 환영합니다! 이슈를 생성하거나 Pull Request를 보내주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
