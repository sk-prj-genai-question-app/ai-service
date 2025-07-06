
from fastapi import APIRouter
from langserve import add_routes
from .chain import chatbot_chain

# FastAPI 라우터 인스턴스 생성
router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
)

# LangServe를 사용하여 챗봇 체인을 라우터에 추가
# 이 함수는 /invoke, /stream, /batch 등의 엔드포인트를 자동으로 생성합니다.
add_routes(
    router,
    chatbot_chain,
)
