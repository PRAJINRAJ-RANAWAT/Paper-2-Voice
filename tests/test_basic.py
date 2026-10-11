"""
Basic tests for Paper to Voice Assistant
"""

import os
import pytest
from src.paper_to_voice.core.config import get_llm
from src.paper_to_voice.utils.pdf_processor import encode_image_to_base64


def test_config_loading_with_api_key(monkeypatch):
    """Test that configuration and LLM load properly when API key is set."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-mock-api-key-12345")
    llm = get_llm()
    assert llm is not None


def test_config_loading_without_api_key(monkeypatch):
    """Test that missing API key raises an explicit ValueError."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_llm()


def test_image_encoding():
    """Test image encoding function existence."""
    assert callable(encode_image_to_base64)
