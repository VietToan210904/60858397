"""Grounded text-only assistance with an application-controlled read-only tool boundary."""

from .tools import ToolController, ToolValidationError
from .validation import evaluate_answer, parse_action, strip_thinking

__all__ = ["ToolController", "ToolValidationError", "evaluate_answer", "parse_action", "strip_thinking"]
