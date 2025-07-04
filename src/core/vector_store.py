import os
import json
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def load_embeddings_from_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadatas = []

    for item in data:
        problem = item["problem"]
        embedding = item["embedding"]
        content = ""
        if problem["question_title_child"]:
            content += problem["question_title_child"] + "\n"
        if problem["question_content"]:
            content += problem["question_content"] + "\n"
        if problem["choice_content"]:
            content += " ".join(problem["choice_content"])

        texts.append(content)
        metadatas.append(problem)

    return texts, metadatas

def save_to_faiss(json_path, faiss_path):
    texts, metadatas = load_embeddings_from_file(json_path)
    embedder = OpenAIEmbeddings()
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]
    vectorstore = FAISS.from_documents(docs, embedder)
    vectorstore.save_local(faiss_path)
    print(f"FAISS 저장 완료: {faiss_path}")

if __name__ == "__main__":
    save_to_faiss("problems_embeddings.json", "faiss_index")
