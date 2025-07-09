from fastapi import APIRouter
from app.chatbot.models.request_schema import QuestionRequest
from app.chatbot.services.chain_selector import select_chain
from app.chatbot.services.chat_history import trim_chat_history, chat_histories

router = APIRouter()

@router.post("/chatbot")
def ask_question(request: QuestionRequest, user_id: str):
    chat_history = chat_histories.get(user_id, "")
    trimmed_history = trim_chat_history(chat_history)

    inputs = {
        "question": request.question,
        "chat_history": trimmed_history
    }

    chain = select_chain(request.question)

    try:
        result = chain.invoke(inputs)
        print("LLM raw output:", result)
    except Exception as e:
        return {
            "answer": str(e),
            "warning": "문제 생성 또는 JSON 파싱에 실패했습니다. 원시 문자열로 반환합니다."
        }

    # 대화 저장
    chat_histories[user_id] = chat_history + f"\n사용자: {request.question}\n어시스턴트: {result}"
    return result
