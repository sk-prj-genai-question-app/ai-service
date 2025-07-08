import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
vectorstore_path = "faiss_index"

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 없습니다. 추가해주세요.")

def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )

    if os.path.exists(vectorstore_path):
        return FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        from document_loader import load_documents
        docs = load_documents()

        # 🔍 Document 체크
        for i, doc in enumerate(docs):
            if not isinstance(doc.page_content, str):
                raise TypeError(f"문서 {i}의 page_content가 문자열이 아닙니다.")
            if not isinstance(doc.metadata, dict):
                raise TypeError(f"문서 {i}의 metadata가 dict가 아닙니다.")
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vectorstore_path)
        return vectorstore
