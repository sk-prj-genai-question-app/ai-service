from fastapi import FastAPI
from routes import chatbot, question_router

app = FastAPI(
    title="JLPT Chatbot API",
    description="JLPT 문법/어휘/독해 기반 질문 응답 및 문제 생성 API",
    version="1.0.0"
)

#app.include_router(question_router.router, prefix="/api")
app.include_router(question_router.router, prefix="/api")