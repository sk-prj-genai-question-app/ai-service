
import json
import random
from fastapi import FastAPI
from pydantic import BaseModel, Field

# RAG 체인 모듈에서 rag_chain 객체를 임포트합니다.
from .rag_chain import rag_chain

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="JLPT 문제 생성 AI 서비스",
    description="LangChain과 RAG를 사용하여 JLPT 문제를 생성하는 API입니다.",
    version="1.0.0",
)

# --- 요청 및 응답 모델 정의 ---

class ProblemRequest(BaseModel):
    level: str = Field(..., description="JLPT 레벨 (예: N1, N2)", example="N1")
    problem_type: str = Field(..., description="문제 유형 (V: 어휘, G: 문법, R: 독해)", example="G")

class Choice(BaseModel):
    number: int
    content: str

class ProblemResponse(BaseModel):
    problem_title_parent: str
    problem_title_child: str | None # None 허용하도록 수정
    problem_content: str | None
    choices: list[Choice]
    answer_number: int
    explanation: str

# --- API 엔드포인트 정의 ---

@app.post("/generate-problem", response_model=ProblemResponse, summary="새로운 JLPT 문제 생성")
def generate_problem(request: ProblemRequest):
    """
    사용자로부터 JLPT 레벨과 문제 유형을 받아, RAG 체인을 통해 새로운 문제를 생성하고 반환합니다.
    """
    # rag_chain에 전달할 질문 생성 (일본어로 쿼리 전달)
    problem_type_jp = {
        "V": "語彙", # 어휘
        "G": "文法", # 문법
        "R": "読解"  # 독해
    }.get(request.problem_type.upper(), "問題")

    # 독해 문제일 경우, 구체적인 주제, 질문 유형, 글 길이를 무작위로 추가
    if request.problem_type.upper() == "R":
        # 주제 목록
        topics = [
            "宗教", "社会", "心理", "技術", "外食", "環境", "歴史", "経済", 
            "芸術", "教育", "科学", "健康", "政治", "文化"
        ]
        selected_topic = random.choice(topics)

        # 질문 유형 목록
        question_types = [
            "本文の内容と一致するものを選ぶ問題",
            "特定の表現の理由を問う問題",
            "指示語が指す内容を問う問題",
            "筆者の主張や意図を問う問題",
            "文章の要旨を把握する問題"
        ]
        selected_question_type = random.choice(question_types)

        # 글 길이 목록
        content_lengths = ["短い文章(約200字)", "中程度の長さの文章(約700字)", "長い文章(約1500字)"]
        selected_length = random.choice(content_lengths)

        query = (f"日本語能力試験（JLPT）の{request.level.upper()}レベルに該当する、"
                 f"「{selected_topic}」に関する読解問題を作成してください。"
                 f"問題のタイプは「{selected_question_type}」で、文章の長さは「{selected_length}」にしてください。"
                 f"作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。"
                 )
    else:
        query = (f"日本語能力試験（JLPT）の{request.level.upper()}"
                 f"レベルに該当する{problem_type_jp}問題を1つ作成してください。"
                 f"作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。"
                 )

    print(f"RAG 체인에 전달할 쿼리: {query}")

    # RAG 체인 실행
    result_str = rag_chain.invoke(query)

    # LLM 출력에서 마크다운 코드 블록 제거 (rag_chain.py에도 있지만, 이중으로 안전장치)
    if result_str.startswith("```json"): 
        result_str = result_str[7:-4].strip()

    try:
        # 문자열 결과를 JSON 객체로 변환
        result_json = json.loads(result_str)
        return result_json
    except json.JSONDecodeError:
        print("JSON 파싱 오류 발생. 원본 출력:", result_str)
        # 여기서는 간단하게 에러 메시지를 포함한 응답을 반환
        return {"error": "Failed to parse LLM output as JSON", "raw_output": result_str}

# 서버 상태 확인을 위한 루트 엔드포인트
@app.get("/", summary="서버 상태 확인")
def read_root():
    return {"status": "JLPT AI Service is running"}

# --- 서버 실행 (테스트용) ---
# 이 파일이 직접 실행될 때 uvicorn 서버를 구동합니다.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
