# 1. 베이스 이미지 설정
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 환경 변수 설정 (Python 로그를 바로 확인하기 위함)
ENV PYTHONUNBUFFERED=1

# 4. 의존성 파일 복사 및 설치
# 먼저 의존성만 설치하여, 코드 변경 시 Docker 빌드 캐시를 활용해 빌드 속도를 높입니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 소스 코드 및 필요 데이터 복사
COPY ./app ./app
COPY ./faiss_index ./faiss_index
COPY ./faiss_index_chatbot ./faiss_index_chatbot

# 6. 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 7. (보안) Non-root 사용자 생성 및 전환
RUN useradd --create-home appuser
USER appuser

# 8. 애플리케이션 실행
# Uvicorn을 사용하여 0.0.0.0 호스트에서 앱을 실행해야 Docker 컨테이너 외부에서 접근 가능합니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
