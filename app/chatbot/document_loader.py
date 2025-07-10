import os
import re
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# jlpt data split : 청크
def split_jlpt_problems(md_text: str, 유형: str):
    docs = []
    if 유형 in ["문법", "어휘"]:
        pattern = r"(##\s*\d+\..*?)(?=##\s*\d+\.|$)"
    elif 유형 == "독해":
        pattern = r"(###.*?)(?=###|$)"
    else:
        return []

    matches = re.findall(pattern, md_text, flags=re.DOTALL)
    for m in matches:
        docs.append(Document(page_content=m.strip(), metadata={}))
    return docs

# jlpt data 메타데이터 설정
def load_documents(base_path=os.path.abspath("data/jlpt_data")):
    docs = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith(".md"):
                continue
            filepath = os.path.join(root, file)
            name_parts = os.path.basename(file).replace(".md", "").split("_")
            if len(name_parts) != 3:
                continue
            level, date, type_code = name_parts
            유형 = {"G": "문법", "V": "어휘", "R": "독해"}.get(type_code.upper(), "기타")

            loader = TextLoader(filepath, encoding="utf-8")
            markdown = loader.load()[0].page_content
            split_docs = split_jlpt_problems(markdown, 유형)

            for doc in split_docs:
                doc.metadata.update({"레벨": level.upper(), "유형": 유형, "파일명": file})
                docs.append(doc)

    return [doc for doc in docs if doc.page_content.strip()]