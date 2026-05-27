"""
# ID: TEST-EMBED-001
# Purpose: Validates embed_texts input guard - empty list must raise ValueError.
#          Full encoding is an integration test (requires model download).
"""

import pytest
from src.embedding import embed_texts


class TestEmbedTexts:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty list"):
            embed_texts([])
