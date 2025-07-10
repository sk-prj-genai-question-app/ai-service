from fastapi import APIRouter
from app.chatbot.models.request_schema import QuestionRequest
from app.chatbot.services.chain_selector import select_chain
from app.chatbot.services.chat_history import trim_chat_history, chat_histories

router = APIRouter()


@router.post("/chatbot")
def ask_question(request: QuestionRequest):
    user_id = request.user_id
    chat_history = chat_histories.get(user_id, "")
    trimmed_history = trim_chat_history(chat_history)

    inputs = {
        "question": request.question,
        "chat_history": trimmed_history
    }

    try:
        chain = select_chain(request.question)
    except ValueError as e: # select_chain에서 유효하지 않은 질문 타입 등으로 오류 발생 시
        return {
            "answer": f"체인을 선택하는 중 오류가 발생했습니다: {e}",
            "warning": "요청을 처리할 수 없습니다. 질문 유형을 확인해주세요."
        }
    except Exception as e: # select_chain에서 예상치 못한 다른 오류 발생 시
        return {
            "answer": f"체인 초기화 중 예상치 못한 오류가 발생했습니다: {e}",
            "warning": "서버 내부 오류가 발생했습니다."
        }

    try:
        result = chain.invoke(inputs)
        print("LLM raw output:", result)
        
    except TypeError as e: # LLM 결과가 예상치 못한 타입일 때 (예: 직렬화 불가능한 객체)
        return {
            "answer": f"LLM 응답 타입 오류: {e}",
            "warning": "LLM이 반환한 데이터를 처리할 수 없습니다."
        }
    except KeyError as e: # LLM 결과 내에서 특정 키를 찾지 못할 때 (내부 체인 구조 문제)
        return {
            "answer": f"LLM 응답 데이터 구조 오류: 필수 정보 누락 ({e})",
            "warning": "LLM 응답의 형식이 올바르지 않습니다."
        }
    except ConnectionError as e: # LLM 서비스와 연결 문제 발생 시
        return {
            "answer": f"LLM 연결 오류: {e}",
            "warning": "현재 LLM 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
        }
    except TimeoutError as e: # LLM 응답 대기 시간 초과 시
        return {
            "answer": f"LLM 응답 시간 초과: {e}",
            "warning": "LLM이 요청 시간 내에 응답하지 않았습니다."
        }
    except Exception as e: # 위에 명시되지 않은 모든 기타 오류
        return {
            "answer": f"알 수 없는 오류 발생: {e}",
            "warning": "문제 생성 또는 처리 중 예상치 못한 오류가 발생했습니다."
        }

    # 대화 저장
    chat_histories[user_id] = chat_history + f"\n사용자: {request.question}\n어시스턴트: {result}"
    return result
