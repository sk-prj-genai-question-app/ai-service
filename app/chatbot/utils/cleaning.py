import re

def clean_json(message):
    try:
        text = message if isinstance(message, str) else str(message)
        text = text.strip()
        text = re.sub(r"^```json\s*|```$", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"(?<!\\)'", '"', text)

        # JSON 시작 위치부터 추출
        json_start = text.find("{")
        if json_start > 0:
            text = text[json_start:]

        return text
    except Exception as e:
        print("clean_json error:", e)
        return str(message)

