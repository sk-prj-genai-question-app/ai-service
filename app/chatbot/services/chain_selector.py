from app.chatbot.routes.llm_chains import prag_chain, grag_chain

def is_generation_request(question: str) -> bool:
    return any(keyword in question for keyword in ["문제", "출제", "만들어", "생성", "개 만들어"])

def select_chain(question: str):
    return prag_chain if is_generation_request(question) else grag_chain