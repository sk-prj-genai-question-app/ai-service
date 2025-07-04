from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def format_problem(meta, content, idx=0):
    title = meta.get("question_title_child", "").strip()
    content_text = content.strip()
    choices = meta.get("choice_content", [])
    choices_text = "\n".join(f"{i+1}) {choice}" for i, choice in enumerate(choices))
    return f"[문제 {idx+1}]\n{title}\n{content_text}\n\n{choices_text}\n{'-'*40}"

def search_similar_questions(query: str, index_path: str = "faiss_index", top_k: int = 5):
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    results = vector_store.similarity_search(query, k=top_k)
    for i, r in enumerate(results):
        print(format_problem(r.metadata, r.page_content, i))

if __name__ == "__main__":
    user_query = input("질문을 입력하세요: ")
    search_similar_questions(user_query)
