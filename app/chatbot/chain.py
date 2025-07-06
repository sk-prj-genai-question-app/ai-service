
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. LLM 정의 (OpenAI gpt-3.5-turbo 사용)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 2. 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Japanese learning assistant. Answer the user's questions politely and concisely in Korean."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# 3. 메모리 구현을 위한 체인 구성
chain = prompt | llm

# 4. 대화 기록을 저장할 딕셔너리 (세션 ID별로 관리)
store = {}

def get_session_history(session_id: str):
    """세션 ID에 해당하는 대화 기록을 가져옵니다."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 5. 대화 기록과 함께 실행 가능한 체인 생성
chatbot_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
