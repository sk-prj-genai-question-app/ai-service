import json
import random
import os
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

# RAG 체인 모듈에서 rag_chain 객체를 임포트합니다.
from .rag_chain import rag_chain

# FastAPI 라우터 인스턴스 생성
router = APIRouter(
    prefix="/problems",
    tags=["problems"],
)

# 백엔드 API URL을 환경 변수에서 가져오거나 기본값 설정
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/problems")

# --- 요청 및 응답 모델 정의 ---

class ProblemRequest(BaseModel):
    level: str = Field(..., description="JLPT 레벨 (예: N1, N2, N3)", example="N1")
    problem_type: str = Field(..., description="문제 유형 (V: 어휘, G: 문법, R: 독해)", example="G")

class Choice(BaseModel):
    id: int | None = None
    number: int
    content: str
    isCorrect: bool

class ProblemDetailResponse(BaseModel):
    id: int | None = None
    level: str
    problem_type: str = Field(..., alias="problemType") # problemType from backend, problem_type in Pydantic
    problem_title_parent: str = Field(..., alias="problemTitleParent")
    problem_title_child: str | None = Field(None, alias="problemTitleChild")
    problem_content: str | None = Field(None, alias="problemContent")
    choices: list[Choice]
    answer_number: int = Field(..., alias="answerNumber")
    explanation: str
    createdAt: datetime | None = Field(None, alias="createdAt")
    updatedAt: datetime | None = Field(None, alias="updatedAt")

class ApiResponseWrapper(BaseModel):
    success: bool
    code: int
    data: ProblemDetailResponse
    message: str
    timestamp: datetime
    requestId: str

# --- API 엔드포인트 정의 ---

@router.post("/generate", response_model=ApiResponseWrapper, summary="새로운 JLPT 문제 생성")
def generate_problem(request: ProblemRequest):
    """
    사용자로부터 JLPT 레벨과 문제 유형을 받아, RAG 체인을 통해 새로운 문제를 생성하고 반환합니다.
    """
    # --- 주제 및 품사 목록 정의 ---
    GENERAL_TOPICS = [
        "人間関係", "仕事", "感情", "自然", "社会", "技術", "健康", "芸術", 
        "経済", "教育", "科学", "歴史", "文化", "環境", "政治", "Eメール", 
        "SNS", "旅行", "食べ物", "スポーツ", "ファッション", "音楽", "映画",
        "文学", "趣味", "ライフスタイル", "国際関係", "法律",
    ]
    PARTS_OF_SPEECH = [
        "名詞", "動詞", "形容詞", "擬音語・擬態語"
    ]

    problem_type_upper = request.problem_type.upper()
    query = ""

    # --- 문제 유형에 따라 동적으로 쿼리 생성 ---

    # 1. 독해 (Reading) 문제
    if problem_type_upper == "R":
        selected_topic = random.choice(GENERAL_TOPICS)
        question_types = [
            "本文の内容と一致するものを選ぶ問題", "特定の表現の理由を問う問題",
            "指示語が指す内容を問う問題", "筆者の主張や意図を問う問題", "文章の要旨を把握する問題"
        ]
        selected_question_type = random.choice(question_types)
        content_lengths = ["短い文章(約200字)", "中程度の長さの文章(約700字)", "長い文章(約1500字)"]
        selected_length = random.choice(content_lengths)

        query = (f"日本語能力試験（JLPT）の{request.level.upper()}レベルに該当する、"
                 f"「{selected_topic}」に関する読解問題を作成してください。"
                 f"問題のタイプは「{selected_question_type}」で、文章の長さは「{selected_length}」にしてください。"
                 f"作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。")

    # 2. 어휘 (Vocabulary) 문제
    elif problem_type_upper == "V":
        selected_topic = random.choice(GENERAL_TOPICS)
        selected_pos = random.choice(PARTS_OF_SPEECH)
        
        query = (f"日本語能力試験（JLPT）の{request.level.upper()}レベルに該当する、"
                 f"「{selected_topic}」をテーマにした「{selected_pos}」の語彙問題を1つ作成してください。"
                 f"作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。")

    # 3. 문법 (Grammar) 및 기타 문제
    else:
        # selected_topic = random.choice(GENERAL_TOPICS)
        problem_type_jp = {"G": "文法"}.get(problem_type_upper, "問題")

        query = (f"日本語能力試験（JLPT）の{request.level.upper()}レベルに該当する、"
                # f"「{selected_topic}」に関連する{problem_type_jp}問題を1つ作成してください。"
                f"{problem_type_jp}問題を1つ作成してください。"
                f"作成する問題の「explanation」フィールドは、必ず韓国語で作成してください。")

    print(f"RAG 체인에 전달할 쿼리: {query}")

    # RAG 체인 실행
    result_str = rag_chain.invoke(query)

    # LLM 출력에서 마크다운 코드 블록 제거 (rag_chain.py에도 있지만, 이중으로 안전장치)
    if result_str.startswith("```json"):
        result_str = result_str[7:-4].strip()

    try:
        # 문자열 결과를 JSON 객체로 변환
        result_json = json.loads(result_str)

        # answer_number를 기준으로 is_correct 필드 추가
        answer_number = result_json.get("answer_number")
        if answer_number:
            for choice in result_json.get("choices", []):
                choice["is_correct"] = (choice.get("number") == answer_number)

        # 응답 데이터에 level과 problem_type 추가
        result_json["level"] = request.level.upper()
        result_json["problem_type"] = request.problem_type.upper()

        # --- Add validation and logging before sending to backend ---
        print(f"JSON to be sent to backend: {json.dumps(result_json, indent=2)}")

        required_fields = ["level", "problem_type", "problem_title_parent", "choices", "answer_number", "explanation"]
        for field in required_fields:
            if field not in result_json or result_json[field] is None:
                print(f"Validation Error: Missing or null required field '{field}' in generated problem JSON.")
                raise HTTPException(status_code=500, detail=f"Generated problem JSON is missing or has null value for required field: {field}")
            # Additional check for empty strings for relevant fields
            if field in ["level", "problem_type", "problem_title_parent", "explanation"] and isinstance(result_json[field], str) and not result_json[field].strip():
                print(f"Validation Error: Required string field '{field}' is empty in generated problem JSON.")
                raise HTTPException(status_code=500, detail=f"Generated problem JSON has empty value for required string field: {field}")

        if not result_json.get("choices"): # choices 리스트가 비어있는 경우도 NotNull 위반으로 간주
            print("Validation Error: 'choices' list is empty in generated problem JSON.")
            raise HTTPException(status_code=500, detail="Generated problem JSON has empty 'choices' list.")

        print(f"백엔드 API로 문제 전송: {BACKEND_API_URL}")
        response = requests.post(BACKEND_API_URL, json=result_json)
        response.raise_for_status() # HTTP 에러 발생 시 예외 발생

        print(f"백엔드 응답 상태 코드: {response.status_code}")
        print(f"백엔드 응답 본문: {response.text}")

        backend_response_data = response.json()

        print(f"문제가 백엔드에 성공적으로 전송되었습니다. 응답 데이터: {backend_response_data}")

        # 프론트엔드에 백엔드 응답의 data 부분을 반환
        return backend_response_data
    except json.JSONDecodeError:
        print("JSON 파싱 오류 발생. 원본 출력:", result_str)
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM output as JSON: {result_str}")
    except requests.exceptions.RequestException as e:
        print(f"백엔드 전송 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send problem to backend: {e}")
