from fastapi import APIRouter
from models.request_schema import QuestionRequest
from vector_store import get_vectorstore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import json

router = APIRouter()

# 1) 벡터스토어 및 retriever
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

# 2) 문서 리스트를 텍스트로 변환
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)
format_docs_runnable = RunnableLambda(format_docs)

# 3) 프롬프트 템플릿
template = """
당신은 JLPT(N1~N3 수준) 일본어 문제를 생성하는 전문가이자 교사입니다.

사용자의 요청을 다음과 같이 처리하십시오:

1. 사용자가 '문제 만들어줘', '문제 생성해줘', 'N3 문법 문제 3개 만들어줘' 등과 같이 문제 생성을 요청하는 경우, 아래 형식에 따라 JSON 형식으로 문제를 생성하세요.
2. 그 외 일반적인 질문에는 일반적인 지식 기반 답변을 하세요.

---

**IMPORTANT INSTRUCTIONS FOR PROBLEM STRUCTURE:**
- problem_title_parent: 문제 세트의 공통 지시문
- problem_title_child: 개별 문제의 문장
- problem_content: 독해 문제용 지문, 문법/어휘 문제의 경우 null로 설정
- choices: 보기 4개
- answer_number: 정답 번호 (1~4)
- explanation: 반드시 한국어로 작성

---

**CRITICAL LANGUAGE RULES**
- JSON의 모든 문제 관련 텍스트는 일본어로 작성
- explanation은 한국어로 작성

사용자 질문: {question}
---
{context}

{chat_history}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question", "context", "chat_history"]
)

# 4) LLM 및 파싱 함수
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
json_parser = StrOutputParser()

def clean_json(text: str) -> str:
    if text.strip().startswith("```json"):
        return text.strip().removeprefix("```json").removesuffix("```").strip()
    return text.strip()
clean_json_runnable = RunnableLambda(clean_json)


# 5) RAG 파이프라인
retrieval_chain = retriever | format_docs_runnable
rag_chain = (
    {
        "context": retrieval_chain,
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough()
    }
    | prompt
    | llm
    | clean_json_runnable
    | json_parser
)

# 6) 문제 생성 요청 여부 판단
def is_generation_request(question: str) -> bool:
    return any(keyword in question for keyword in ["문제", "출제", "만들어", "생성", "개 만들어"])

# 7) FastAPI 라우터
@router.post("/ask")
def ask_question(request: QuestionRequest):
    chat_history = ""  # 필요시 확장
    inputs = {
        "question": request.question,
        "chat_history": chat_history
    }

    if is_generation_request(request.question):
        try:
            result = rag_chain.invoke(inputs)
            return {"answer": result}
        except Exception as e:
            return {
                "answer": str(e),
                "warning": "문제 생성 또는 JSON 파싱에 실패했습니다. 원시 문자열로 반환합니다."
            }
    else:
        result = llm.invoke(request.question)
        return {"answer": result}
