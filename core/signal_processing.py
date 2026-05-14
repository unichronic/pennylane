"""Small helper for pulling the portfolio rating out of PM output.

The rating is already in the rendered markdown, so this just keeps the old
``SignalProcessor.process_signal(text)`` shape around.
"""

from __future__ import annotations

from typing import Any

from agents.rating import parse_rating


class SignalProcessor:
    """Pull the rating out of PM text."""

    def __init__(self, quick_thinking_llm: Any = None):
        # Older callers still pass an llm here. We just ignore it now.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return one of Buy / Overweight / Hold / Underweight / Sell."""
        return parse_rating(full_signal)
