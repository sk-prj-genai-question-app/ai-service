import os
import json
from src.chains.markdown_parser import parse_markdown


def process_all_md_files(folder_path):
    all_problems = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            problems = parse_markdown(md_text, filename)  # filename 인자 추가
            all_problems.extend(problems)
    return all_problems


if __name__ == "__main__":
    folder = "./data/scraped_data"  # 마크다운 파일들이 있는 경로
    problems = process_all_md_files(folder)
    print(f"총 {len(problems)}개의 문제를 파싱했습니다.")

    # JSON 파일로 저장
    with open("problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False, indent=2)
