from fastapi import APIRouter
from models.request_schema import QuestionRequest, JLPTProblem, GenerationProblem
from vector_store import get_vectorstore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
import re

router = APIRouter()

# 벡터스토어 및 retriever
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

# chat history token 수 제한
MAX_TURNS = 700

def trim_chat_history(history: str, max_turns=MAX_TURNS) -> str:
    turns = history.strip().split("\n")
    trimmed = turns[-(max_turns * 2):]
    return "\n".join(trimmed)


# context 제한
def format_docs_limited(docs: list[Document], max_length: int = 1500) -> str:
    combined = ""
    for doc in docs:
        if len(combined) + len(doc.page_content) > max_length:
            break
        combined += doc.page_content + "\n\n"
    return combined.strip()
format_docs_runnable = RunnableLambda(lambda docs: format_docs_limited(docs, max_length=1500))


# 프롬프트 템플릿
template_problem = """
당신은 JLPT(N1~N3 수준) 일본어 문제를 생성하는 전문가이자 교사입니다.

사용자의 요청을 다음과 같이 처리하십시오:

1. 사용자가 '문제 만들어줘', '문제 생성해줘', 'N3 문법 문제 3개 만들어줘' 등과 같이 문제 생성을 요청하는 경우, 아래 형식에 따라 JSON 형식으로 문제를 생성하세요.
2. 그 외 일반적인 질문에는 일반적인 지식 기반 답변을 하세요.

**CRITICAL LANGUAGE RULES**
- JSON의 모든 문제 관련 텍스트는 일본어로 작성
- explanation은 한국어로 작성

{{
  "is_problem": true,
  "problem_title_parent": "string",
  "problem_title_child": "string",
  "problem_content": "string (can be null if not applicable)",
  "choices": [
    {{"number": 1, "content": "string"}},
    {{"number": 2, "content": "string"}},
    {{"number": 3, "content": "string"}},
    {{"number": 4, "content": "string"}}
  ],
  "answer_number": "integer (from 1 to 4)",
  "explanation": "string (detailed explanation of why the answer is correct and others are not)"
}}


사용자 질문: {question}
---
{context}

{chat_history}

위에 제시된 JSON 스키마에 맞춰 응답을 생성하십시오. 다른 어떤 추가적인 텍스트나 설명을 포함하지 마세요.
"""


template_generation = """
당신은 일본어 관련 응답을 해주는 전문가이자 교사입니다.

사용자는 일본어와 관련된 질문을 합니다. 아래 형식에 따라 JSON 형식으로 답변을 생성해주세요.
사용자 질문: {question}

이전에 했던 질의 : {chat_history}

JSON 응답의 모든 문자열은 큰따옴표(")로 감싸야 합니다. 작은따옴표(')를 쓰지 마세요.
반드시 아래와 같은 JSON 객체 형태로 응답하세요:
{{
    "is_problem": false,
    "answer": "..."
}}

"""

# 문제 생성 관련 프롬프트
problemPrompt = PromptTemplate(
    template=template_problem,
    input_variables=["question", "context", "chat_history"]
)

# 일반 질문 관련 프롬프트
generationPrompt = PromptTemplate(
    template=template_generation,
    input_variables=["question", "chat_history"]
)

# LLM 및 파싱 함수
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
json_parser = JsonOutputParser()

# llm 응답 정리
def clean_json(message):
    try:
        text = message if isinstance(message, str) else str(message)
        text = text.strip()
        text = re.sub(r"^```json\s*|```$", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"(?<!\\)'", '"', text)

        # JSON 시작 위치부터 추출
        json_start = text.find("{")
        if json_start > 0:
            text = text[json_start:]

        return text
    except Exception as e:
        print("clean_json error:", e)
        return str(message)
clean_json_runnable = RunnableLambda(clean_json)


# 파서 설정
problem_pydantic_parser = PydanticOutputParser(pydantic_object=JLPTProblem)
general_pydantic_parser = PydanticOutputParser(pydantic_object=GenerationProblem)
str_parser = StrOutputParser()


# RAG 파이프라인
retrieval_chain = RunnableLambda(lambda inputs: retriever.invoke(inputs["question"])) | format_docs_runnable
prag_chain = (
    {
        "context": retrieval_chain,
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | problemPrompt
    | llm
    | str_parser
    | clean_json_runnable
    | problem_pydantic_parser
)

grag_chain = (
    {
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | generationPrompt
    | llm
    | str_parser
    | clean_json_runnable
    | general_pydantic_parser
)

# 문제 생성 요청 여부 판단
def is_generation_request(question: str) -> bool:
    return any(keyword in question for keyword in ["문제", "출제", "만들어", "생성", "개 만들어"])

# 서버가 살아있는 동안 유지되는 메모리 기반 chat_history 저장소
chat_histories = {}


# FastAPI 라우터
@router.post("/ask")
def ask_question(request: QuestionRequest, user_id: str): # user_id는 추후에 바꿀 예정
    
    # user_id는 클라이언트가 매 요청에 함께 보내야 함
    chat_history = chat_histories.get(user_id, "")
    trimmed_history = trim_chat_history(chat_history)

    inputs = {
        "question": request.question,
        "chat_history": trimmed_history
    }

    if is_generation_request(request.question):
        try:
            result = prag_chain.invoke(inputs)
            print("LLM raw output:", result)
        except Exception as e:
            return {
                "answer": str(e),
                "warning": "문제 생성 또는 JSON 파싱에 실패했습니다. 원시 문자열로 반환합니다."
            }
    else:
        result = grag_chain.invoke(inputs)
        
    # 대화 누적 (간단한 형식, 필요하면 포맷 조절 가능)
    chat_histories[user_id] = chat_history + f"\n사용자: {request.question}\n어시스턴트: {result}"
    return result