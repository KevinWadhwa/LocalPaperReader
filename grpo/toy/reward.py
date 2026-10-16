import re

answer_pattern = r"<answer>(.*?)</answer>"
num_pattern = r"-?\d+"

def parse_answer(text):
    """
    Return the final answer between the tags, or none if not present.
    """
    matches = re.findall(answer_pattern, text)
    if not matches:
        return None

    inner = matches[-1].strip()

    if not re.fullmatch(num_pattern, inner):
        return None

    return int(inner)

def reward(text, answer):
    """
    If correct then 1.0 otherwise 0.0.
    """
    parsed = parse_answer(text)
    if parsed is None:
        return 0.0
    elif parsed == answer:
        return 1.0
    else:
         return 0.0

