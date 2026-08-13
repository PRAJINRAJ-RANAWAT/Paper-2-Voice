"""
Basic tests for Paper to Voice Assistant
"""

import os

import pytest

from src.paper_to_voice.core.config import get_llm
from src.paper_to_voice.utils.pdf_processor import encode_image_to_base64


def test_config_loading():
    """Test that get_llm function exists."""
    assert callable(get_llm)


def test_image_encoding():
    """Test that image encoding function exists."""
    assert callable(encode_image_to_base64)


if __name__ == "__main__":
    test_config_loading()
    test_image_encoding()
    print("🎉 Basic tests completed!")
