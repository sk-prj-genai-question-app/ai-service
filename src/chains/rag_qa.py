from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
import os

# LangSmith 비활성화
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "no-key"  # LangSmith 연결 차단

def create_stuff_chain():
    # 임베딩 모델 및 벡터스토어 로드
    embedder = OpenAIEmbeddings()
    vector_store = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    # LLM 설정 (GPT-3.5 Turbo로 변경 - 비용 절감)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # 한국어 프롬프트 (출력은 일본어)
    prompt_template = """
    당신은 JLPT 자격증 시험의 출제자 입니다. 
    JLPT {level} {question_type} 문제를 {num}개 생성하세요.
    - 형식: 
    問題[번호]. [문제 문장]
    1) [선택지1]
    2) [선택지2]
    3) [선택지3]
    4) [선택지4]
    정답: [번호]
    
    - 규칙:
    1. 각 문제마다 완전히 다른 문법 패턴 사용
    2. 질문과 선택지는 무조건 문법적으로 정확하고 자연스러운 일본어
    3. 정답 번호는 1~4에 고르게 분포
    4. 문제 문장은 실제 생활에서 사용되는 자연스러운 표현
    5. {level} 난이도에 맞춤
    6. 기출문제 패턴 참고: {context}

    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "level", "question_type", "num"]
    )
    
    return vector_store, StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        document_variable_name="context"
    )

def generate_questions(vector_store, chain, level: str, q_type: str, num: int):
    docs = vector_store.as_retriever().invoke(f"{level} {q_type} 過去問")
    result = chain.invoke({
        "input_documents": docs,
        "level": level,
        "question_type": q_type,
        "num": num
    })
    return result["output_text"]

if __name__ == "__main__":
    try:
        vs, chain = create_stuff_chain()
        print(" JLPT 문제 생성기 준비 완료")
        
        while True:
            print("\n" + "="*40)
            print(" JLPT 문제 생성기 ".center(40, "="))
            print("1. N1  2. N2  3. N3  0. 종료")
            level = input("레벨 선택: ").strip()
            
            if level == "0":
                break
                
            if level not in ["1", "2", "3"]:
                print("※ 1~3 사이 숫자 입력")
                continue
                
            print("\n1. 文法(문법)  2. 語彙(어휘)  3. 読解(독해)")
            q_type = input("유형 선택: ").strip()
            
            if q_type not in ["1", "2", "3"]:
                print("※ 1~3 사이 숫자 입력")
                continue
                
            try:
                num = int(input("문제 수: "))
                num = max(1, min(30, num))
            except ValueError:
                print("※ 숫자 입력")
                continue
                
            levels = {"1": "N1", "2": "N2", "3": "N3"}
            types = {"1": "文法", "2": "語彙", "3": "読解"}
            
            print(f"\n[{levels[level]} {types[q_type]} 문제 {num}개 생성]")
            result = generate_questions(vs, chain, levels[level], types[q_type], num)
            print(result)
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("※ faiss_index 폴더와 .env 파일을 확인해주세요")