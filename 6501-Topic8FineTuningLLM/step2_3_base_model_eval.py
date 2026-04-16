from __future__ import annotations

import argparse
import os
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker import types

from sql_finetuning_utils import (
    BASE_MODEL,
    NUM_TEST_EXAMPLES,
    RANDOM_SEED,
    format_prompt,
    load_dataset,
    split_dataset,
)
from sql_matches import sql_matches


ENV_PATH = Path('.env')
DEFAULT_EVAL_EXAMPLES = 5
MAX_NEW_TOKENS = 128


def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    prompt = format_prompt(context=context, question=question)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    sampling_params = types.SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop=['\n', '<|end_of_text|>'],
    )
    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=1,
    ).result()
    return tokenizer.decode(result.sequences[0].tokens).strip()


def evaluate_examples(test_data: list[dict], limit: int, base_model: str) -> float:
    load_dotenv(dotenv_path=ENV_PATH)
    if not os.getenv('TINKER_API_KEY'):
        raise RuntimeError('TINKER_API_KEY is missing. Check your .env file.')

    print(f'Initializing Tinker service client for base model: {base_model}')
    service_client = tinker.ServiceClient()

    print('Creating sampling client...')
    sampling_client = service_client.create_sampling_client(base_model=base_model)

    print('Fetching tokenizer from sampling client...')
    tokenizer = sampling_client.get_tokenizer()
    print('Tokenizer ready.')

    num_correct = 0
    eval_slice = test_data[:limit]

    print(f'Evaluating base model: {base_model}')
    print(f'Number of held-out examples to evaluate now: {len(eval_slice)}')
    print()

    for index, example in enumerate(eval_slice, start=1):
        generated_sql = sample_from_model(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            context=example['context'],
            question=example['question'],
        )
        is_match = sql_matches(
            generated=generated_sql,
            expected=example['answer'],
            schema=example['context'],
        )
        num_correct += int(is_match)

        print(f'=== Example {index}/{len(eval_slice)} ===')
        print(f"Question:  {example['question']}")
        print(f"Schema:    {example['context']}")
        print(f"Expected:  {example['answer']}")
        print(f"Generated: {generated_sql}")
        print(f"Match:     {is_match}")
        print()

    accuracy = num_correct / len(eval_slice)
    print(f'Accuracy: {num_correct}/{len(eval_slice)} = {accuracy:.1%}')
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate the base Tinker model on held-out text-to-SQL examples.',
    )
    parser.add_argument(
        '--eval-examples',
        type=int,
        default=DEFAULT_EVAL_EXAMPLES,
        help='How many held-out test examples to evaluate right now.',
    )
    parser.add_argument(
        '--base-model',
        default=BASE_MODEL,
        help='Base model name to sample from.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_examples <= 0:
        raise ValueError('--eval-examples must be positive.')

    data = load_dataset()
    _, test_data = split_dataset(
        data=data,
        num_test_examples=NUM_TEST_EXAMPLES,
        seed=RANDOM_SEED,
    )
    evaluate_examples(
        test_data=test_data,
        limit=min(args.eval_examples, len(test_data)),
        base_model=args.base_model,
    )


if __name__ == '__main__':
    main()
