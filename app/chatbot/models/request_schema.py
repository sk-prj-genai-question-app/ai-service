from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str
    user_id: str = Field(alias="userId")

class Choice(BaseModel):
    number: int
    content: str

class JLPTProblem(BaseModel):
    is_problem: bool
    level: str
    problem_type: str
    problem_title_parent: str
    problem_title_child: str
    problem_content: str
    choices: list[Choice]
    answer_number: int
    explanation: str

class GenerationProblem(BaseModel):
    is_problem: bool
    answer: str  = Field(description="질문에 대한 답변")