from fastapi import APIRouter
from models.request_schema import QuestionRequest
from vector_store import get_vectorstore
from langchain.chains import RetrievalQA, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json

router = APIRouter()
llm = ChatOpenAI(model="gpt-4o")
vectorstore = get_vectorstore()

# 문제 생성용 프롬프트 템플릿
template = """
당신은 JLPT(N1~N3 수준) 일본어 문제를 생성하는 전문가이자 교사입니다.

사용자의 요청을 다음과 같이 처리하십시오:

1. 사용자가 '문제 만들어줘', '문제 생성해줘', 'N3 문법 문제 3개 만들어줘' 등과 같이 **문제 생성을 요청하는 경우**, 아래 형식에 따라 **JSON 형식으로 문제를 생성**하세요.
2. 그 외 일반적인 질문(예: "N3 문법 예시 알려줘", "어휘 뜻 알려줘")에는 일반적인 지식 기반 답변을 하세요.

---

**IMPORTANT INSTRUCTIONS FOR PROBLEM STRUCTURE:**
- **problem_title_parent**: 문제 세트의 공통 지시문
- **problem_title_child**: 개별 문제의 문장
- **problem_content**: 독해 문제용 지문, 문법/어휘 문제의 경우 `null`로 설정
- **choices**: 보기 4개 (1~4번)
- **answer_number**: 정답 번호 (1~4)
- **explanation**: 반드시 **한국어**로 작성

---

**CRITICAL LANGUAGE RULES**
- JSON의 모든 문제 관련 텍스트는 **일본어**로 작성
- `explanation`은 **한국어**로 작성

사용자 질문: {question}
---
{context}
"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# 문제 생성 요청 여부 판단
def is_generation_request(question: str) -> bool:
    return any(keyword in question for keyword in ["문제", "출제", "만들어", "생성", "개 만들어"])

# 질문 처리 분기
def run_chain(question: str):
    if is_generation_request(question):
        res = qa_chain.invoke({"query": question, "context": ""})
        if isinstance(res, dict) and "result" in res:
            result = res
        else:
            result = {"result": res}
    else:
        direct_prompt = PromptTemplate(template="{question}", input_variables=["question"])
        direct_chain = LLMChain(llm=llm, prompt=direct_prompt)
        result_text = direct_chain.invoke({"question": question})
        result = {"result": result_text}
    return result


@router.post("/ask")
def ask_question(request: QuestionRequest):
    result = run_chain(request.question)
    raw_result = result["result"]

    # 문제 생성 요청인 경우 (JSON 문자열을 반환하는 경우)
    if is_generation_request(request.question):
        # 마크다운 블록 제거 (예: ```json ... ```)
        if isinstance(raw_result, str) and raw_result.strip().startswith("```json"):
            cleaned = raw_result.strip().removeprefix("```json").removesuffix("```").strip()
        else:
            cleaned = raw_result.strip()

        try:
            parsed_json = json.loads(cleaned)
            return {"answer": parsed_json}
        except json.JSONDecodeError:
            return {
                "answer": raw_result,
                "warning": "JSON 파싱에 실패했습니다. 원시 문자열로 반환합니다."
            }
        
    return {"answer": result["result"]}