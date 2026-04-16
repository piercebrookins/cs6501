from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
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
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_LOG_EVERY = 1
DEFAULT_SAVE_NAME = 'sql-smoke-run'
DEFAULT_MAX_TRAIN_EXAMPLES = 256


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


def tensor_values(value) -> list[float]:
    if hasattr(value, 'tolist'):
        return value.tolist()
    return list(value)


def compute_mean_nll(loss_fn_outputs, batch: list[types.Datum]) -> float:
    total_weighted_nll = 0.0
    total_weight = 0.0

    for output, datum in zip(loss_fn_outputs, batch):
        logprobs = tensor_values(output['logprobs'])
        weights = tensor_values(datum.loss_fn_inputs['weights'])
        for logprob, weight in zip(logprobs, weights):
            total_weighted_nll += (-float(logprob)) * float(weight)
            total_weight += float(weight)

    if total_weight == 0:
        return math.nan
    return total_weighted_nll / total_weight


def build_processed_training_data(train_data: list[dict], tokenizer, max_train_examples: int | None) -> list[types.Datum]:
    if max_train_examples is not None:
        train_data = train_data[:max_train_examples]
    return [process_example(example, tokenizer) for example in train_data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a LoRA text-to-SQL model with Tinker.',
    )
    parser.add_argument('--base-model', default=BASE_MODEL, help='Base model to fine-tune.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help='Adam learning rate.',
    )
    parser.add_argument(
        '--max-train-examples',
        type=int,
        default=DEFAULT_MAX_TRAIN_EXAMPLES,
        help='Cap the number of training examples for smoke runs. Use 0 or omit cap for full run.',
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=DEFAULT_LOG_EVERY,
        help='Print loss every N optimizer steps.',
    )
    parser.add_argument(
        '--save-name',
        default=DEFAULT_SAVE_NAME,
        help='Checkpoint name to use with save_weights_for_sampler.',
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=32,
        help='LoRA rank for the training client.',
    )
    return parser.parse_args()


async def create_training_client_with_heartbeat(service_client, base_model: str, lora_rank: int):
    print(f'Creating Tinker training client for base model: {base_model}')
    task = asyncio.create_task(
        service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=lora_rank,
        )
    )

    wait_seconds = 0
    while not task.done():
        if wait_seconds > 0:
            print(
                f'Still waiting for Tinker to allocate the training client '
                f'({wait_seconds}s elapsed)...'
            )
        await asyncio.sleep(10)
        wait_seconds += 10

    training_client = await task
    print('Training client ready.')
    return training_client


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')
    if args.epochs <= 0:
        raise ValueError('--epochs must be positive.')
    if args.learning_rate <= 0:
        raise ValueError('--learning-rate must be positive.')
    if args.log_every <= 0:
        raise ValueError('--log-every must be positive.')

    load_dotenv(dotenv_path=ENV_PATH)
    if not os.getenv('TINKER_API_KEY'):
        raise RuntimeError('TINKER_API_KEY is missing. Check your .env file.')

    service_client = tinker.ServiceClient()
    training_client = asyncio.run(
        create_training_client_with_heartbeat(
            service_client=service_client,
            base_model=args.base_model,
            lora_rank=args.lora_rank,
        )
    )

    print('Fetching tokenizer from training client...')
    tokenizer = training_client.get_tokenizer()
    print('Tokenizer ready.')

    print('Loading and splitting dataset...')
    data = load_dataset()
    train_data, test_data = split_dataset(
        data=data,
        num_test_examples=NUM_TEST_EXAMPLES,
        seed=RANDOM_SEED,
    )
    print(f'Total training examples available: {len(train_data)}')
    print(f'Held-out test examples: {len(test_data)}')

    max_train_examples = None if args.max_train_examples == 0 else args.max_train_examples
    print('Processing training examples into Tinker Datum objects...')
    processed_train = build_processed_training_data(
        train_data=train_data,
        tokenizer=tokenizer,
        max_train_examples=max_train_examples,
    )
    if not processed_train:
        raise ValueError('No training examples selected.')

    print(f'Prepared training examples for this run: {len(processed_train)}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.learning_rate}')
    print()

    step = 0
    for epoch in range(args.epochs):
        print(f'=== Epoch {epoch + 1}/{args.epochs} ===')
        random.Random(RANDOM_SEED + epoch).shuffle(processed_train)

        for batch_start in range(0, len(processed_train), args.batch_size):
            batch = processed_train[batch_start: batch_start + args.batch_size]
            if not batch:
                continue

            step += 1
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn='cross_entropy')
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=args.learning_rate)
            )

            fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_future.result()
            train_mean_nll = compute_mean_nll(fwd_bwd_result.loss_fn_outputs, batch)

            if step % args.log_every == 0:
                print(
                    f'Step {step}: '
                    f'batch_size={len(batch)} '
                    f'train_mean_nll={train_mean_nll:.4f}'
                )

    print()
    print(f'Saving sampler weights as: {args.save_name}')
    sampler_path = training_client.save_weights_for_sampler(args.save_name).result().path
    print(f'Saved sampler path: {sampler_path}')


if __name__ == '__main__':
    main()
