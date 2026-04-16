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
    format_prompt_and_completion,
    load_dataset,
    split_dataset,
)


ENV_PATH = Path('.env')
DEFAULT_PREVIEW_EXAMPLES = 2
DEFAULT_PREVIEW_TOKENS = 40


def process_example(example: dict, tokenizer) -> types.Datum:
    prompt, completion = format_prompt_and_completion(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion_tokens = tokenizer.encode(f' {completion}\n', add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    shifted_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            'target_tokens': target_tokens,
            'weights': shifted_weights,
        },
    )


def preview_datum(datum: types.Datum, tokenizer, max_rows: int) -> None:
    input_tokens = datum.model_input.to_ints()
    target_tokens = datum.loss_fn_inputs['target_tokens'].tolist()
    weights = datum.loss_fn_inputs['weights'].tolist()

    print(f"{'Input':<20} {'Target':<20} {'Weight':<6}")
    print('-' * 52)
    for idx, (inp, tgt, weight) in enumerate(zip(input_tokens, target_tokens, weights)):
        if idx >= max_rows:
            print('...')
            break
        input_text = repr(tokenizer.decode([int(inp)]))
        target_text = repr(tokenizer.decode([int(tgt)]))
        print(f'{input_text:<20} {target_text:<20} {int(weight):<6}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare text-to-SQL training examples as Tinker Datum objects.',
    )
    parser.add_argument(
        '--base-model',
        default=BASE_MODEL,
        help='Base model name used to fetch the tokenizer from Tinker.',
    )
    parser.add_argument(
        '--preview-examples',
        type=int,
        default=DEFAULT_PREVIEW_EXAMPLES,
        help='How many processed training examples to preview.',
    )
    parser.add_argument(
        '--preview-tokens',
        type=int,
        default=DEFAULT_PREVIEW_TOKENS,
        help='How many token rows to print per previewed example.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_examples <= 0:
        raise ValueError('--preview-examples must be positive.')
    if args.preview_tokens <= 0:
        raise ValueError('--preview-tokens must be positive.')

    load_dotenv(dotenv_path=ENV_PATH)
    if not os.getenv('TINKER_API_KEY'):
        raise RuntimeError('TINKER_API_KEY is missing. Check your .env file.')

    print(f'Creating sampling client for tokenizer: {args.base_model}')
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.base_model)
    tokenizer = sampling_client.get_tokenizer()
    print('Tokenizer ready.')

    data = load_dataset()
    train_data, test_data = split_dataset(
        data=data,
        num_test_examples=NUM_TEST_EXAMPLES,
        seed=RANDOM_SEED,
    )
    print(f'Training examples available: {len(train_data)}')
    print(f'Held-out test examples: {len(test_data)}')
    print()

    preview_slice = train_data[:args.preview_examples]
    processed = [process_example(example, tokenizer) for example in preview_slice]

    for index, (example, datum) in enumerate(zip(preview_slice, processed), start=1):
        prompt, completion = format_prompt_and_completion(example)
        print(f'=== Processed training example {index}/{len(processed)} ===')
        print('Prompt:')
        print(prompt)
        print()
        print('Completion:')
        print(completion)
        print()
        preview_datum(datum, tokenizer, max_rows=args.preview_tokens)
        print()


if __name__ == '__main__':
    main()
