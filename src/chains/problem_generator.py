from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rag_qa import create_stuff_chain, generate_questions
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"  # 필요시 4로 변경 가능
)

prompt_template = PromptTemplate.from_template("""
아래 조건에 맞는 {count}개의 문제를 만들어줘:

- 자격증: JLPT
- 레벨: {level}
- 유형: {category}
- 형식: 객관식 4지선다

출력 형식 예:
1. 문제 내용
A. 보기1
B. 보기2
C. 보기3
D. 보기4
정답: A
""")

chain = LLMChain(llm=llm, prompt=prompt_template)

vs, chain = create_stuff_chain()

def generate_problems_rag(level, category, count):
    return generate_questions(vs, chain, level, category, count)
