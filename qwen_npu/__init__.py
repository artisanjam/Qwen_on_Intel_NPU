"""Qwen3-8B on Intel NPU: engine, watchdog and CLI modules.

Submodules are imported explicitly; nothing is loaded on package
import, so consumers can pull only what they need (e.g. a server
importing qwen_npu.session without the CLI).
"""
