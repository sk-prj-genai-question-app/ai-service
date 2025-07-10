MAX_TURNS = 700
chat_histories = {}

def trim_chat_history(history: str, max_turns=MAX_TURNS) -> str:
    turns = history.strip().split("\n")
    trimmed = turns[-(max_turns * 2):]
    return "\n".join(trimmed)