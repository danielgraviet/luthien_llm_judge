from openai import OpenAI
import numpy as np
import random
from dotenv import load_env
from src.rubrics import code as CODE_RUBRIC
load_env()

_PAIRWISE_TEMPLATE = """{rubric}

Task:
{prompt}

Candidate A:
{resp_a}

Candidate B:
{resp_b}

Compare A and B using the rubric.
Answer only:
"A" if A is better,
"B" if B is better,
"Tie" if they are equivalent.
Also explain your reasoning briefly.
"""

client = OpenAI()


def judge_pointwise(prompt: str, response: str, rubric: str = CODE_RUBRIC):
    system = "You are a precise, unbiased evaluator."
    user = f"{rubric}\n\nPrompt:\n{prompt}\n\nCandidate Response:\n{response}\n\nProvide a score from 1 to 5 and a short explanation."
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0
    )
    return result.choices[0].message.content


def judge_pairwise(prompt: str, resp_a: str, resp_b: str, rubric: str = CODE_RUBRIC):
    system = "You are an impartial expert evaluator."
    user = _PAIRWISE_TEMPLATE.format(
        rubric=rubric,
        prompt=prompt,
        resp_a=resp_a,
        resp_b=resp_b,
    )
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0
    )
    return result.choices[0].message.content


def judge_hybrid(prompt, resp_a, resp_b):
    pw = judge_pairwise(prompt, resp_a, resp_b)
    s_a = judge_pointwise(prompt, resp_a)
    s_b = judge_pointwise(prompt, resp_b)

    # Parse scores (simplified demo)
    def extract_score(text):
        for t in text.split():
            if t.strip().isdigit():
                return int(t)
        return None

    score_a, score_b = extract_score(s_a), extract_score(s_b)

    if "A" in pw:
        pair_choice = "A"
    elif "B" in pw:
        pair_choice = "B"
    else:
        pair_choice = "Tie"

    # Weighted decision
    if pair_choice == "A" and (score_a or 0) >= (score_b or 0):
        winner = "A"
    elif pair_choice == "B" and (score_b or 0) > (score_a or 0):
        winner = "B"
    else:
        winner = "Tie"

    return {
        "pairwise": pair_choice,
        "scores": (score_a, score_b),
        "final": winner
    }

