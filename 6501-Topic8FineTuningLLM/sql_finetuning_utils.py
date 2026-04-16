from __future__ import annotations

import json
import random
from pathlib import Path


DATA_PATH = Path('sql_create_context_v4.json')
BASE_MODEL = 'meta-llama/Llama-3.2-1B'
NUM_TEST_EXAMPLES = 200
RANDOM_SEED = 42
REQUIRED_KEYS = {'question', 'context', 'answer'}


def load_dataset(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')

    with path.open(encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError('Expected top-level JSON array of examples.')
    if not data:
        raise ValueError('Dataset is empty.')

    missing = REQUIRED_KEYS - set(data[0].keys())
    if missing:
        raise ValueError(f'Dataset examples are missing keys: {sorted(missing)}')

    return data


def split_dataset(
    data: list[dict],
    num_test_examples: int = NUM_TEST_EXAMPLES,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict]]:
    if len(data) <= num_test_examples:
        raise ValueError('Need more examples than the requested test split size.')

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    test_data = shuffled[:num_test_examples]
    train_data = shuffled[num_test_examples:]
    return train_data, test_data


def format_prompt(context: str, question: str) -> str:
    return f'Table schema:\n{context}\nQuestion: {question}\nSQL:'


def format_prompt_and_completion(example: dict) -> tuple[str, str]:
    prompt = format_prompt(context=example['context'], question=example['question'])
    completion = example['answer']
    return prompt, completion


def preview_example(example: dict, index: int) -> None:
    print(f'--- Sample {index} ---')
    print(f"Question: {example['question']}")
    print(f"Context:  {example['context']}")
    print(f"Answer:   {example['answer']}")
    print()
