from qasper.reward import ABSTAIN

MAX_PROMPT_TOKENS = 4000

SYSTEM_PROMPT = f"""You answer questions about a research paper using only the numbered paragraphs provided.

1. First, copy 1 to 3 passages word for word from the paragraphs that support your answer. Each passage must be 5 to 50 words, written as <quote p="N">passage</quote>, where N is the paragraph's number.
2. Then give a short answer as <answer>answer</answer>. Answer yes/no questions with exactly Yes or No.
3. If the paragraphs do not contain the answer, write only: <answer>{ABSTAIN}</answer>

Tables and figures are not shown, only their captions."""

EXAMPLE_CONTEXT = """Title: Parsing Low-Resource Languages

[P1] (Introduction) We study dependency parsing for languages with little annotated data.

[P2] (Experiments) We evaluate on the Universal Dependencies treebanks for Basque and Welsh, reporting labelled attachment score."""

FEW_SHOT = [
    {"role": "user", "content": f"{EXAMPLE_CONTEXT}\n\nQuestion: Which languages do they evaluate on?"},
    {"role": "assistant", "content": '<quote p="2">We evaluate on the Universal Dependencies treebanks for Basque and Welsh</quote>\n<answer>Basque and Welsh</answer>'},
    {"role": "user", "content": f"{EXAMPLE_CONTEXT}\n\nQuestion: What hardware did they train on?"},
    {"role": "assistant", "content": f"<answer>{ABSTAIN}</answer>"},
]


def build_messages(context, question, few_shot=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        messages += FEW_SHOT
    messages.append({"role": "user", "content": f"{context}\n\nQuestion: {question}"})
    return messages


def format_prompt(tok, context, question, few_shot=False):
    return tok.apply_chat_template(build_messages(context, question, few_shot),
                                   tokenize=False, add_generation_prompt=True)