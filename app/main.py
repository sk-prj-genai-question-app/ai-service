
from fastapi import FastAPI
from .problem_generator.router import router as problem_generator_router

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="JLPT AI 서비스",
    description="LangChain과 RAG를 사용하여 JLPT 문제를 생성하고, AI 챗봇 기능을 제공하는 API입니다.",
    version="1.1.0",
)

# 문제 생성기 라우터 포함
app.include_router(problem_generator_router)

# 여기에 나중에 챗봇 라우터를 추가할 수 있습니다.
# from .chatbot.router import router as chatbot_router
# app.include_router(chatbot_router)

# 서버 상태 확인을 위한 루트 엔드포인트
@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "JLPT AI Service is running"}

# --- 서버 실행 (테스트용) ---
# 이 파일이 직접 실행될 때 uvicorn 서버를 구동합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
