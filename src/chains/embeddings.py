import json
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def create_embeddings_batch(problems_file):
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )

    with open(problems_file, "r", encoding="utf-8") as f:
        problems = json.load(f)

    texts = []
    enriched_problems = []

    for problem in problems:
        text = ""
        if problem["question_title_child"]:
            text += problem["question_title_child"] + "\n"
        if problem["question_content"]:
            text += problem["question_content"] + "\n"
        if problem["choice_content"]:
            text += " ".join(problem["choice_content"])
        texts.append(text)
        enriched_problems.append(problem)

    print(f"총 {len(texts)}개의 문제 텍스트를 배치 임베딩 중...")

    vectors = embedder.embed_documents(texts)

    results = []
    for problem, vector in zip(enriched_problems, vectors):
        results.append({
            "problem": problem,
            "embedding": vector
        })

    return results

if __name__ == "__main__":
    output_file = "embeddings.json"
    problems_file = "problems.json"

    embeddings = create_embeddings_batch(problems_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    print(f"{len(embeddings)}개의 문제 임베딩 완료! → {output_file} 저장됨")

    with open("problems_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)