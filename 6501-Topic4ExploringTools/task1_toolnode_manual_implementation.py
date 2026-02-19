"""Task 1 wrapper: manual ToolNode implementation.

This file exists to satisfy portfolio naming requirements while keeping logic DRY.
Actual implementation lives in `toolnode_example.py`.
"""

from toolnode_example import main


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
