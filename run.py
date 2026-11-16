"""Convenience shim: python run.py starts the interactive CLI.

The canonical invocation is python -m qwen_npu, which runs the
package's __main__.py. This script keeps the documented
historical invocation working unchanged.
"""

from qwen_npu.cli import main

if __name__ == "__main__":
    main()
