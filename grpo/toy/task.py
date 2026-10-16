import random

system_prompt = (
    "You are a mathematician. Work through the problem step by step. "
    "Then give the final number, as digits no commas on one line, inside <answer></answer>."
)

def make_problem(rng, difficulty):
    if difficulty == 1:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"What is {a} + {b}?", a + b
    if difficulty == 2:
        a, b = rng.randint(10, 99), rng.randint(2, 9)
        return f"What is {a} * {b}?", a * b
    if difficulty == 3:
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(10, 99)
        return f"What is {a} * {b} + {c}?", a * b + c
    if difficulty == 4:
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(100, 999)
        return f"What is {a} * {b} - {c}?", a * b - c
    raise ValueError(f"unknown difficulty: {difficulty}")

def build_messages(question):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

def format_prompt(tokenizer, question):
    return tokenizer.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )

def make_dataset(seed, n, difficulty):
    rng = random.Random(seed)
    return [make_problem(rng, difficulty) for _ in range(n)]

