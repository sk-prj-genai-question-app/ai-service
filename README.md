# 🧠 JLPT 문제 생성 학습 도우미 - AI 서비스

## 1. 🚀 프로젝트 소개

이 프로젝트는 "생성형 AI를 통한 JLPT 문제 생성 학습 도우미" 웹 애플리케이션의 AI 서비스 부분입니다. JLPT 문제 생성, 사용자 질문에 대한 챗봇 응답 등 핵심 AI 기능을 제공합니다.

## 2. 🛠️ 기술 스택

- **언어**: Python
- **프레임워크**: FastAPI
- **AI/ML 라이브러리**: LangChain, OpenAI, Google GenAI, FAISS
- **기타**: Uvicorn, python-dotenv, pydantic, unstructured, markdown

## 3. ✨ 주요 기능

- **JLPT 문제 생성**: AI 모델을 활용하여 JLPT 시험 유형에 맞는 문제 생성
- **챗봇 기능**: 사용자 질문에 대한 AI 기반 답변 제공
- **FAISS 기반 벡터 검색**: 효율적인 정보 검색 및 활용을 위한 FAISS 인덱스 관리
- **다양한 AI 모델 연동**: OpenAI, Google GenAI 등 다양한 LLM(Large Language Model) 연동 지원

## 4. ⚙️ 환경 설정

프로젝트를 로컬에서 실행하기 위해 다음 환경 변수 설정이 필요합니다. 프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 채워주세요.

```
# OpenAI API Key (필요시)
OPENAI_API_KEY=your_openai_api_key_here

# Google GenAI API Key (필요시)
GOOGLE_API_KEY=your_google_api_key_here
```

## 5. ▶️ 실행 방법

1.  **환경 변수 설정**: 위 4번 항목을 참조하여 `.env` 파일을 설정합니다.
2.  **의존성 설치**: 다음 명령어를 실행하여 필요한 Python 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```
3.  **애플리케이션 실행**: 다음 명령어로 FastAPI 애플리케이션을 실행합니다.
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
4.  **접속**: 애플리케이션은 기본적으로 `http://localhost:8000` 포트에서 실행됩니다. API 문서는 `http://localhost:8000/docs` 에서 확인하실 수 있습니다.

## 6. 📖 API 문서

FastAPI는 자동으로 OpenAPI(Swagger UI) 문서를 생성합니다. 애플리케이션 실행 후 `http://localhost:8000/docs` 에서 상세한 API 명세를 확인하고 테스트할 수 있습니다. 또는, docs 리포지토리에서 확인할 수 있습니다.

## 7. 📄 라이선스

이 프로젝트는 MIT License를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
