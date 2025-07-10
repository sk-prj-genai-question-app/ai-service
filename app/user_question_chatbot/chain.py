import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,  
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# .env 파일에서 환경 변수 로드
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY .env 파일에 없습니다. 추가해주세요.")

# 1. LLM 정의 (문제 생성에 사용된 모델과 동일하게 gemini-1.5-pro-latest 사용)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",  # 또는 "llama3-70b-8192", "gemma-7b-it"
#     temperature=0.7,
# )

# 2. 채팅 프롬프트 템플릿 정의
# MessagesPlaceholder를 사용하여 이전 대화 기록을 동적으로 삽입합니다.
chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
            You are a helpful Japanese language tutor specializing in JLPT reading comprehension.
            Your goal is to provide clear and concise answers to user questions.
            You should act as a supportive tutor for JLPT studies, assisting with general questions related to Japanese language learning, JLPT exams, or providing explanations for various Japanese concepts.
            Always respond in Korean.

            --- Provided JLPT Problem Information ---
            Problem ID: {problem_id}
            Level: {problem_level}
            Type: {problem_type}
            Parent Title: {problem_title_parent}
            Child Title: {problem_title_child}
            Problem Content:
            {problem_content}

            Choices:
            {problem_choices_formatted}
            Correct Answer Number: {problem_answer_number}
            Explanation:
            {problem_explanation}
            ----------------------------------

            # Instructions for Answering:
            Based on the "Problem Content" and "Provided JLPT Problem Information" above, answer the user's "question".
            Carefully analyze the "Problem Content" to identify the author's main argument or theme, and any supporting examples or historical contexts mentioned.
            Do not make up information that is not present in the "Problem Content" or "Explanation".
            If the "Problem Content" does not contain enough information to answer, state that clearly.
            Focus on extracting the information directly from the given text.

            User Question: {question}
            """,
            input_variables=[ # <--- input_variables를 명시적으로 추가했습니다.
                "problem_id", "problem_level", "problem_type",
                "problem_title_parent", "problem_title_child", "problem_content",
                "problem_choices_formatted", "problem_answer_number", "problem_explanation",
                "question"
            ]
        ),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# 3. 채팅 체인 조립
def create_chatbot_chain():
    # RunnablePassthrough를 사용하여 입력 변수를 체인으로 전달
    return (
        RunnablePassthrough.assign(
            problem_id=lambda x: x.get('problem_id'),
            problem_level=lambda x: x.get('problem_level'),
            problem_type=lambda x: x.get('problem_type'),
            problem_title_parent=lambda x: x.get('problem_title_parent'),
            problem_title_child=lambda x: x.get('problem_title_child'),
            problem_content=lambda x: x.get('problem_content') or "",
            problem_choices_formatted=lambda x: "\n".join([
                f" {choice['number']}. {choice['content']}" # <--- 여기서 ['number'] 와 ['content'] 로 변경
                for choice in x.get('problem_choices', [])
            ]),
            problem_answer_number=lambda x: x.get('problem_answer_number'),
            problem_explanation=lambda x: x.get('problem_explanation') or "",
            question=lambda x: x.get('question'),
        )
        | chat_template
        | llm
        | StrOutputParser()
    )

# 챗봇 체인 인스턴스 생성
chatbot_chain = create_chatbot_chain()

if __name__ == "__main__":
    # 테스트용 예제 사용법 (이 부분도 변경된 입력 변수들을 포함해야 함)
    print("챗봇 체인을 테스트합니다.")

    # 예시 문제 데이터 (실제 데이터 구조에 맞춰야 함)
    sample_problem_data = {
        "problem_id": 1,
        "problem_level": "N2",
        "problem_type": "R",
        "problem_title_parent": "次の文章を読んで、後の問いに答えなさい。",
        "problem_title_child": "第２段落にある「これ」は何を指すか。最も適切なものを、1・2・3・4から一つ選びなさい。",
        "problem_content": "近年の外食産業は、多様化する顧客のニーズに応えるべく、様々なサービスを展開している。...", # 실제 본문 내용
        "problem_choices": [
            {"id": 1, "number": 1, "content": "外食産業が社会全体のwell-beingに貢献する存在へと進化していること", "is_correct": True},
            {"id": 2, "number": 2, "content": "外食産業が多様なサービスを提供し顧客の利便性を高めていること", "is_correct": False}
        ],
        "problem_answer_number": 1,
        "problem_explanation": "1) 외식 산업이 사회 전체의 웰빙에 기여하는 존재로 진화하고 있는 것: 정답입니다. ..."
    }

    current_question_1 = "제2단락에 나오는 'これ(코레)'가 정확히 무엇을 지칭하는지 자세히 설명해 주세요."
    chat_history_1 = []

    print(f"\n--- 첫 번째 질문: {current_question_1} ---")
    response_1 = chatbot_chain.invoke({
        "question": current_question_1,
        "chat_history": chat_history_1,
        **sample_problem_data # 문제 데이터를 체인 입력에 포함
    })
    print(f"AI 응답: {response_1}")

    # 두 번째 질문 (이전 대화 기록 포함)
    current_question_2 = "그럼 이 글 전체의 주제는 무엇이라고 생각하나요?"
    chat_history_2 = [
        HumanMessage(content=current_question_1),
        AIMessage(content=response_1)
    ]

    print(f"\n--- 두 번째 질문: {current_question_2} (이전 대화 포함) ---")
    response_2 = chatbot_chain.invoke({
        "question": current_question_2,
        "chat_history": chat_history_2,
        **sample_problem_data # 문제 데이터를 체인 입력에 계속 포함
    })
    print(f"AI 응답: {response_2}")
    print("------------------------")