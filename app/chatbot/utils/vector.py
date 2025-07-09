from vector_store import get_vectorstore

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever()

def format_docs_limited(docs, max_length: int = 1500) -> str:
    combined = ""
    for doc in docs:
        if len(combined) + len(doc.page_content) > max_length:
            break
        combined += doc.page_content + "\n\n"
    return combined.strip()
