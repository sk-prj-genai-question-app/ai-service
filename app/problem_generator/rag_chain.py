import os
import json
import re
from dotenv import load_dotenv

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 1. .env 파일에서 환경 변수 로드
load_dotenv()

# LangChain 캐싱 설정
set_llm_cache(InMemoryCache())
print("LangChain InMemoryCache가 활성화되었습니다.")

# GOOGLE_API_KEY가 있는지 확인
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 없습니다. 추가해주세요.")

# --- 벡터 저장소 준비 (캐싱 및 일괄 처리 로직) ---

# 2. FAISS 인덱스 파일 경로 정의
vectorstore_path = "faiss_index"

# 3. 임베딩 모델 정의
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 4. 벡터 저장소 로드 또는 생성
if os.path.exists(vectorstore_path):
    # 인덱스 파일이 존재하면, 로컬에서 바로 로드
    print(f"'{vectorstore_path}'에서 기존 벡터 저장소를 로드합니다.")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
else:
    # 인덱스 파일이 없으면, 문서를 처리하고 새로 생성
    print(f"'{vectorstore_path}'를 찾을 수 없습니다. 새로운 벡터 저장소를 생성합니다.")
    
    # --- JLPT 데이터 로딩 및 분할 ---
    jlpt_data_path = "data/jlpt_data"
    jlpt_loader = DirectoryLoader(jlpt_data_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True)
    jlpt_docs = jlpt_loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    jlpt_splits = []
    for doc in jlpt_docs:
        jlpt_splits.extend(markdown_splitter.split_text(doc.page_content))
    print(f"JLPT 데이터에서 {len(jlpt_splits)}개의 청크를 생성했습니다.")

    # --- Article 데이터 로딩 및 분할 ---
    article_data_path = "data/article_data"
    article_loader = DirectoryLoader(article_data_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True)
    article_docs = article_loader.load()

    article_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 청크 크기 (조절 가능)
        chunk_overlap=200, # 청크 간 중복 (조절 가능)
        length_function=len,
        is_separator_regex=False,
    )
    article_splits = article_text_splitter.split_documents(article_docs)
    print(f"Article 데이터에서 {len(article_splits)}개의 청크를 생성했습니다.")

    # --- 모든 청크 결합 ---
    all_splits = jlpt_splits + article_splits
    print(f"총 {len(all_splits)}개의 청크를 임베딩합니다.")

    # 일괄 처리를 위한 텍스트 및 메타데이터 준비
    texts = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata for doc in all_splits]
    batch_size = 100 # 한 번에 처리할 문서 수 (API 제한에 따라 조절 가능)

    print(f"{len(texts)}개의 텍스트를 {batch_size}개씩 일괄 처리하여 벡터 저장소를 생성합니다.")

    # 첫 번째 배치로 벡터 저장소 초기화
    vectorstore = FAISS.from_texts(texts[:batch_size], embeddings, metadatas=metadatas[:batch_size])

    # 나머지 배치들을 순차적으로 추가
    for i in range(batch_size, len(texts), batch_size):
        vectorstore.add_texts(texts[i:i + batch_size], metadatas=metadatas[i:i + batch_size])
        print(f"{i + batch_size} / {len(texts)} 처리 완료...")

    vectorstore.save_local(vectorstore_path)
    print(f"벡터 저장소를 '{vectorstore_path}'에 저장했습니다.")

# 5. 검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# --- 프롬프트 및 LLM 체인 ---

# 6. 프롬프트 템플릿 정의
template = """
You are an expert JLPT tutor. Create a new, original Japanese Language Proficiency Test (JLPT) problem based on the provided context examples.
The generated problem must be in the same style and level as the context.
Output your response in a valid JSON format.

JSON Structure:
{{
  "problem_title_parent": "string (general instruction, e.g., '次の文の（ ）に入れるのに最もよいものを、1・2・3・4から一つ選びなさい。')",
  "problem_title_child": "string (specific question statement)",
  "problem_content": "string (for reading comprehension, null otherwise)",
  "choices": [
    {{"number": 1, "content": "string"}},
    {{"number": 2, "content": "string"}},
    {{"number": 3, "content": "string"}},
    {{"number": 4, "content": "string"}}
  ],
  "answer_number": "integer (1-4)",
  "explanation": "string (detailed explanation in Korean)"
}}

Language Requirements:
- 'explanation' field MUST be in Korean.
- All other fields (problem_title_parent, problem_title_child, problem_content, choices content) MUST be in Japanese.

Problem Generation Rules:
- Generate a completely new and original problem. Do NOT repeat previous problems.
- Strictly follow the 'QUESTION' specifications for topic, question type, and content length.
- For Reading Comprehension, select a suitable passage from CONTEXT matching the requested topic and content length. Create a question matching the requested question type.
- Ensure the problem is appropriate for the specified JLPT level.

Explanation Format:
- Each choice's explanation MUST be: "{{choice_number}}) {{choice_content}}: {{reason}}".
- Each explanation for a choice MUST be separated by a newline character (
).

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT (in valid JSON format):
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 7. LLM 정의
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.85)

# --- 출력 파서 및 체인 조립 ---

def clean_json_output(text: str) -> str:
    """LLM 출력에서 마크다운 코드 블록을 제거하여 순수 JSON 문자열을 추출합니다."""
    # "```json"으로 시작하고 "```"으로 끝나는 블록을 찾습니다.
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() # 매치되지 않으면 원본 텍스트를 반환

# 8. LangChain Expression Language (LCEL)를 사용하여 RAG 체인 정의
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def format_docs_limited(docs, max_length: int = 3000) -> str:
    combined = ""
    for doc in docs:
        if len(combined) + len(doc.page_content) > max_length:
            break
        combined += doc.page_content + "\n\n"
    return combined.strip()

rag_chain = (
    { "context": retriever | format_docs_limited, "question": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
    | clean_json_output # 정제 함수를 체인에 추가
)

# --- 테스트용 예제 사용법 ---
if __name__ == "__main__":
    import re # 정규 표현식 모듈 임포트
    # 이 스크립트를 직접 실행하여 체인을 테스트합니다.
    print("RAG Chain을 테스트합니다. N1 어휘 문제 1개를 생성하는 요청을 실행합니다...")

    test_query = ("日本語能力試験（JLPT）のN1"
                 "レベルに該当する、語彙問題を1つ作成してください。"
                 "作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。"
                 )
    
    # 테스트 쿼리를 체인에 전달
    result_str = rag_chain.invoke(test_query)
    
    print("\n--- 생성된 문제 (JSON) ---")
    try:
        # 문자열 결과를 JSON으로 파싱하여 예쁘게 출력
        result_json = json.loads(result_str)
        print(json.dumps(result_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("오류: LLM이 유효한 JSON을 반환하지 않았습니다.")
        print(result_str)
    print("------------------------")

