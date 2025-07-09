# app/chatbot/router.py (수정)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from .chain import chatbot_chain # 챗봇 체인 임포트
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage # LangChain 메시지 타입 임포트
import json

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
    user_question_id: int
    question: str
    chat_history: List[Dict[str, str]]

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
    print(f"\n--- FastAPI Received Request (Parsed) ---")
    print(request.model_dump_json(indent=2)) 
    print("----------------------------------------\n")

    print(f"Received chat request for UserQuestion ID: {request.user_question_id}")

    # problem_content가 None이 아닐 때만 출력
    if request.problem_content is not None:
        print(f"Received problem content (first 100 chars): {request.problem_content[:100]}...")
    else:
        print("Received problem content: None")


    langchain_chat_history: List[BaseMessage] = []
    for msg in request.chat_history:
        if 'role' in msg and 'content' in msg:
            if msg['role'].lower() == "user": # 딕셔너리 키로 접근
                langchain_chat_history.append(HumanMessage(content=msg['content'])) # 딕셔너리 키로 접근
            elif msg['role'].lower() == "ai" or msg['role'].lower() == "assistant":
                langchain_chat_history.append(AIMessage(content=msg['content']))
            elif msg['role'].lower() == "system":
                langchain_chat_history.append(SystemMessage(content=msg['content']))
            else:
                print(f"Warning: Unknown message role in history: {msg['role']}")
        else:
            print(f"Warning: Invalid chat history message format encountered: {msg}")
            
    # ProblemChoice 객체의 리스트를 원래의 딕셔너리 리스트로 다시 변환
    # chatbot_chain에 전달될 최종 입력 데이터를 준비합니다.
    # None 값을 빈 문자열이나 빈 리스트로 처리하여 체인 내부 오류 방지
    chain_input_data = {
        "question": request.question,
        "chat_history": langchain_chat_history, # LangChain 형식으로 변환된 대화 기록
        "user_question_id": request.user_question_id,
        "problem_id": request.problem_id,
        "problem_level": request.problem_level if request.problem_level is not None else "",
        "problem_type": request.problem_type if request.problem_type is not None else "",
        "problem_title_parent": request.problem_title_parent if request.problem_title_parent is not None else "",
        "problem_title_child": request.problem_title_child if request.problem_title_child is not None else "",
        "problem_content": request.problem_content if request.problem_content is not None else "",
        # ProblemChoice 객체의 리스트를 LangChain 체인에 전달하기 위해 model_dump()로 딕셔너리 리스트로 다시 변환
        "problem_choices": [choice.model_dump() for choice in request.problem_choices] if request.problem_choices is not None else [],
        "problem_answer_number": request.problem_answer_number,
        "problem_explanation": request.problem_explanation if request.problem_explanation is not None else ""
    }

    debug_chain_input = chain_input_data.copy()
    if 'chat_history' in debug_chain_input:
        debug_chain_input['chat_history'] = [
            msg.model_dump() for msg in langchain_chat_history
        ]

    print(f"\n--- Final Chain Input before Invocation (router.py) ---")
    print(json.dumps(debug_chain_input, indent=2, ensure_ascii=False)) # chain_input -> chain_input_data
    print("------------------------------------------------------\n")


    try:
        response_content = chatbot_chain.invoke(chain_input_data)

        return ChatResponse(response=response_content)
    except Exception as e:
        print(f"챗봇 처리 중 오류 발생: {e}")
        # 오류 발생 시 클라이언트에게 500 Internal Server Error 반환
        raise HTTPException(status_code=500, detail=f"AI chatbot internal error: {e}")
