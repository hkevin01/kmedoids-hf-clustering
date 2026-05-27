"""
# ID: TEST-DATA-001
# Purpose: Unit tests for src/data_loader.py - validates cleaning logic and
#          error handling without requiring network access.
"""

import re
import pytest
from src.data_loader import _clean_text


class TestCleanText:
    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert _clean_text("foo   bar") == "foo bar"

    def test_collapses_newlines(self):
        result = _clean_text("line1\n\nline2")
        assert result == "line1 line2"

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_no_change_needed(self):
        assert _clean_text("hello world") == "hello world"
