from typing import Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from chains.problem_generator import generate_problems
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="생성형 AI를 이용한 JLPT 자격증 공부")

class ProblemRequest(BaseModel):
    level: Literal["n1", "n2", "n3"]
    category: Literal["문법", "어휘", "독해"]
    count: Optional[int] = 30

@app.post("/generate-problems")
def generate_problems_endpoint(request: ProblemRequest):
    result = generate_problems(request.level, request.category, request.count)
    return {"problems": result}
