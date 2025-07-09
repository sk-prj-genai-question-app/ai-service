import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.chatbot.document_loader import load_documents

load_dotenv()
vectorstore_path = "faiss_index_chatbot"

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다. 추가해주세요.")

def get_vectorstore():
    embeddings = OpenAIEmbeddings()

    if os.path.exists(vectorstore_path):
        print(f"기존 벡터 저장소 '{vectorstore_path}' 로드")
        return FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("문서를 불러옵니다...")
        docs = load_documents()

        if not docs:
            raise ValueError("load_documents()가 빈 리스트를 반환했습니다.")

        # 유효한 문서만 필터링
        docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        if not docs:
            raise ValueError("유효한 page_content가 있는 문서가 없습니다.")

        # 형 검사
        for i, doc in enumerate(docs):
            if not isinstance(doc.page_content, str):
                raise TypeError(f"문서 {i}의 page_content가 문자열이 아닙니다.")
            if not isinstance(doc.metadata, dict):
                raise TypeError(f"문서 {i}의 metadata가 dict가 아닙니다.")

        print(f"총 {len(docs)}개의 문서에서 FAISS 벡터를 생성합니다...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vectorstore_path)
        print(f"벡터 저장소가 '{vectorstore_path}'에 저장되었습니다.")
        return vectorstore