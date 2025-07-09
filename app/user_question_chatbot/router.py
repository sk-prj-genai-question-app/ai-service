# app/chatbot/router.py (수정)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from .chain import chatbot_chain # 챗봇 체인 임포트
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage # LangChain 메시지 타입 임포트

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/user_question_chatbot", # 이 라우터의 모든 엔드포인트는 /chatbot으로 시작
    tags=["User Question Chatbot"], # Swagger UI 그룹핑을 위한 태그
)

# --- 요청 및 응답 모델 정의 ---

# 대화 메시지 모델 (클라이언트로부터 받을 메시지 형식)
class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (예: user, ai, system)")
    content: str = Field(..., description="메시지 내용")

# ProblemChoice 모델
class ProblemChoice(BaseModel):
    id: int
    number: int
    content: str
    is_correct: bool

# 챗봇 요청 모델
class ChatRequest(BaseModel):
    user_question_id: int = Field(..., description="현재 대화가 속한 UserQuestion의 ID")
    question: str = Field(..., description="사용자의 현재 질문")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="이전 대화 기록 (선택 사항)")

    # --- 문제 관련 정보 필드 추가 ---
    problem_id: Optional[int] = None
    problem_level: Optional[str] = None 
    problem_type: Optional[str] = None
    problem_title_parent: Optional[str] = None
    problem_title_child: Optional[str] = None
    problem_content: Optional[str] = None 
    problem_choices: Optional[List[ProblemChoice]] = None
    problem_answer_number: Optional[int] = None
    problem_explanation: Optional[str] = None
# 챗봇 응답 모델
class ChatResponse(BaseModel):
    response: str = Field(..., description="AI의 응답 메시지")


# --- API 엔드포인트 정의 ---

@router.post("/ask", response_model=ChatResponse, summary="챗봇에게 질문하기")
async def ask_chatbot(request: ChatRequest):
    """
    챗봇에게 질문을 하고 AI의 응답을 받습니다. 이전 대화 기록을 포함하여 컨텍스트를 유지할 수 있습니다.
    """
    try:
        print("\n--- Received Request Data ---")
        print(request.model_dump_json(indent=2))
        print("---------------------------\n")
        print(f"Received chat request for UserQuestion ID: {request.user_question_id}")

        # problem_content가 None이 아닐 때만 출력
        if request.problem_content is not None:
            print(f"Received problem content (first 100 chars): {request.problem_content[:100]}...")
        else:
            print("Received problem content: None")


        langchain_chat_history: List[BaseMessage] = []
        for msg in request.chat_history:
            if msg.role.lower() == "user":
                langchain_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role.lower() == "ai" or msg.role.lower() == "assistant":
                langchain_chat_history.append(AIMessage(content=msg.content))
            elif msg.role.lower() == "system":
                langchain_chat_history.append(SystemMessage(content=msg.content))
            else:
                print(f"Warning: Unknown message role in history: {msg.role}")
                continue

        # ProblemChoice 객체의 리스트를 원래의 딕셔너리 리스트로 다시 변환
        # chatbot_chain에 전달될 최종 입력 데이터를 준비합니다.
        # None 값을 빈 문자열이나 빈 리스트로 처리하여 체인 내부 오류 방지
        chain_input_data = {
            "question": request.question,
            "chat_history": langchain_chat_history,
            "problem_id": request.problem_id,
            "problem_level": request.problem_level,
            "problem_type": request.problem_type,
            "problem_title_parent": request.problem_title_parent,
            "problem_title_child": request.problem_title_child,
            "problem_content": request.problem_content if request.problem_content is not None else "", # None이면 빈 문자열
            "problem_choices": [choice.model_dump() for choice in request.problem_choices] if request.problem_choices is not None else [], # None이면 빈 리스트
            "problem_answer_number": request.problem_answer_number,
            "problem_explanation": request.problem_explanation if request.problem_explanation is not None else "" # None이면 빈 문자열
        }


        # 챗봇 체인 실행
        response_content = chatbot_chain.invoke(chain_input_data)

        return ChatResponse(response=response_content)
    except Exception as e:
        print(f"챗봇 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")