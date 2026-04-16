from pathlib import Path
import os
import sys

from dotenv import load_dotenv


def main() -> int:
    env_path = Path(__file__).with_name('.env')
    loaded = load_dotenv(dotenv_path=env_path)
    key = os.getenv('TINKER_API_KEY')

    print(f'.env found: {env_path.exists()}')
    print(f'.env loaded: {loaded}')
    print(f'TINKER_API_KEY present: {bool(key)}')

    if key:
        redacted = f'{key[:6]}...{key[-4:]}' if len(key) >= 12 else '***set***'
        print(f'TINKER_API_KEY preview: {redacted}')

    try:
        import tinker  # noqa: F401
        print('Tinker import works: True')
    except Exception as exc:
        print(f'Tinker import works: False ({exc})')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
