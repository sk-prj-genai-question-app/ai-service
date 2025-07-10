from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from app.chatbot.models.request_schema import JLPTProblem, GenerationProblem
from app.chatbot.utils.vector import get_retriever, format_docs_limited
from app.chatbot.utils.cleaning import clean_json

# 프롬프트 템플릿
# 문제 생성 질문 프롬프트 템플릿
template_problem = """
You are an expert and teacher specializing in generating JLPT (N1-N3 level) Japanese questions.

Process user requests as follows:

1. When the user asks for problem generation (e.g., '문제 만들어줘', '문제 생성해줘', 'N3 문법 문제 3개 만들어줘'), generate problems in JSON format according to the structure below.
2. For other general questions, provide general knowledge-based answers.

**CRITICAL LANGUAGE RULES**
- All problem-related text within the JSON must be in Japanese.
- The 'explanation' must be in Korean.
- 'level' must be one of "N1", "N2", or "N3".
- 'problem_type' must be one of "G" (Grammar), "R" (Reading), or "V" (Vocabulary).

{{
  "is_problem": true,
  "level": "N3", // One of "N1", "N2", "N3"
  "problem_type": "G", // One of "G", "R", "V"
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

User question: {question}
---
{context}

{chat_history}

Generate the response strictly following the JSON schema provided above. Do not include any additional text or explanations.
"""

# 일반 질문 프롬프트 템플릿
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
