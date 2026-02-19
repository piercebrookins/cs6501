"""Task 2 wrapper: create_react_agent-based implementation.

This file exists to satisfy portfolio naming requirements while keeping logic DRY.
Actual implementation lives in `react_agent_example.py`.
"""

from react_agent_example import main


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
