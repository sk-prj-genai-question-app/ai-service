import re
from typing import List, Dict, Any

def get_question_type_from_filename(filename: str) -> str:
    """파일명에서 문제 유형 추출"""
    type_map = {
        '_G.md': '문법',
        '_V.md': '어휘',
        '_R.md': '독해'
    }
    for suffix, q_type in type_map.items():
        if filename.endswith(suffix):
            return q_type
    return '기타'

def parse_markdown(md_text: str, filename: str = "") -> List[Dict[str, Any]]:
    """  
    Args:
        md_text: 파싱할 마크다운 텍스트
        filename: 문제 유형 추출을 위한 파일명(옵션)
    
    Returns:
        문제 딕셔너리의 리스트
    """
    question_type = get_question_type_from_filename(filename)
    problems = []
    current_parent = None  # # 큰제목
    current_child = None   # ## 소제목
    current_content = []   # ### 본문
    current_choices = []   # 선택지

    # 선택지 패턴
    choice_pattern = re.compile(r"^\s*[-*]\s*(\d+[.)])\s*(.*)$")

    for line in md_text.splitlines():
        stripped = line.strip()
        
        # 레벨1 헤더 (#)
        if line.startswith('# ') and '##' not in line[:3]:
            _save_problem_if_exists(problems, current_parent, current_child, 
                                   current_content, current_choices, question_type)
            current_parent = stripped[2:].strip()
            current_child = None
            current_content = []
            current_choices = []
            
        # 레벨2 헤더 (##)
        elif line.startswith('## ') and '###' not in line[:4]:
            _save_problem_if_exists(problems, current_parent, current_child, 
                                  current_content, current_choices, question_type)
            current_child = stripped[3:].strip()
            current_content = []
            current_choices = []
            
        # 레벨3 헤더 (###) → 문제 본문 시작
        elif line.startswith('### '):
            current_content.append(stripped[4:].strip())
            
        # 레벨4+ 헤더 (####) → 무시
        elif line.startswith('####'):
            continue
            
        # 선택지 처리
        elif choice_pattern.match(stripped):
            _, choice_text = choice_pattern.match(stripped).groups()
            current_choices.append(choice_text.strip())
            
        # 일반 텍스트 (문제 본문에 추가)
        elif stripped and current_content is not None:
            current_content.append(stripped)
    
    # 마지막 문제 저장
    _save_problem_if_exists(problems, current_parent, current_child, 
                           current_content, current_choices, question_type)
    
    return problems

def _save_problem_if_exists(problems: List[Dict], 
                           parent: str, 
                           child: str, 
                           content: List[str], 
                           choices: List[str], 
                           q_type: str) -> None:
    """문제 데이터가 존재할 경우에만 저장"""
    if not parent:
        return
        
    # 내용 정제
    clean_content = '\n'.join(content).strip()
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
    
    # 선택지가 있는 경우에만 문제로 인정
    if choices:
        problems.append({
            "question_title_parent": parent,
            "question_title_child": child,
            "question_content": clean_content,
            "choice_content": choices,
            "question_type": q_type,
        })