from __future__ import annotations

from sql_finetuning_utils import (
    DATA_PATH,
    NUM_TEST_EXAMPLES,
    RANDOM_SEED,
    load_dataset,
    preview_example,
    split_dataset,
)


NUM_SAMPLES_TO_PRINT = 3


def main() -> None:
    data = load_dataset()
    print(f'Total examples: {len(data)}')
    print(f'Example keys: {sorted(data[0].keys())}')
    print()

    for i, example in enumerate(data[:NUM_SAMPLES_TO_PRINT], start=1):
        preview_example(example, i)

    train_data, test_data = split_dataset(
        data=data,
        num_test_examples=NUM_TEST_EXAMPLES,
        seed=RANDOM_SEED,
    )

    print(f'Training examples: {len(train_data)}')
    print(f'Test examples: {len(test_data)}')
    print()

    print('First held-out test example after shuffle:')
    preview_example(test_data[0], 1)


if __name__ == '__main__':
    main()
