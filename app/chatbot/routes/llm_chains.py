from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from app.chatbot.models.request_schema import JLPTProblem, GenerationProblem
from app.chatbot.utils.vector import get_retriever, format_docs_limited
from app.chatbot.utils.cleaning import clean_json

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
You are a Japanese language expert and teacher.
Refer to the previous conversation only when necessary.
When the user asks a question about the Japanese language, respond in Korean.

However, if you need to include Japanese words or expressions to explain something clearly, feel free to use Japanese where appropriate.

Always answer in the following JSON format. Ensure the JSON is strictly valid and can be directly parsed without errors. Do not include any extra text outside the JSON.

```json
{{
  "is_problem": false,
  "answer": "..."
}}

- All strings must use double quotes (\\"), and single quotes (') should not be used.
- Escape all double quotes inside string values with a backslash (\\").
- Refer to the previous conversation only if it's relevant.

User question: {question}

Previous conversation: {chat_history}
"""

problemPrompt = PromptTemplate(
    template=template_problem,
    input_variables=["question", "context", "chat_history"]
)
generationPrompt = PromptTemplate(
    template=template_generation,
    input_variables=["question", "chat_history"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
retriever = get_retriever()
format_docs_runnable = RunnableLambda(lambda docs: format_docs_limited(docs, max_length=1500))
clean_json_runnable = RunnableLambda(clean_json)
str_parser = StrOutputParser()


# 파서
problem_pydantic_parser = PydanticOutputParser(pydantic_object=JLPTProblem)
general_pydantic_parser = PydanticOutputParser(pydantic_object=GenerationProblem)

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
