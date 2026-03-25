"""Tests for experiment.adapters.llmservingsim."""

from __future__ import annotations

import os

import pytest

from experiment.adapters.llmservingsim import LLMServingSimAdapter, MODEL_MAP


# ---------------------------------------------------------------------------
# Tests: MODEL_MAP constant
# ---------------------------------------------------------------------------


class TestModelMap:
    def test_model_map_llama(self):
        """Test Llama model mapping strips -Instruct suffix."""
        assert MODEL_MAP["meta-llama/Llama-3.1-8B-Instruct"] == "meta-llama/Llama-3.1-8B"

    def test_model_map_mixtral(self):
        """Test Mixtral model mapping remains unchanged."""
        assert MODEL_MAP["mistralai/Mixtral-8x7B-v0.1"] == "mistralai/Mixtral-8x7B-v0.1"

    def test_model_map_coverage(self):
        """Test MODEL_MAP contains exactly 2 supported models."""
        assert len(MODEL_MAP) == 2


# ---------------------------------------------------------------------------
# Tests: adapter basics
# ---------------------------------------------------------------------------


class TestLLMServingSimAdapterBasics:
    def test_name(self, tmp_path):
        """Adapter name should be 'llmservingsim'."""
        # Create a fake main.py so the constructor doesn't raise
        main_py = tmp_path / "main.py"
        main_py.write_text("")
        adapter = LLMServingSimAdapter(str(tmp_path))
        assert adapter.name == "llmservingsim"

    def test_init_stores_absolute_path(self, tmp_path):
        """Constructor should store absolute path to LLMServingSim dir."""
        main_py = tmp_path / "main.py"
        main_py.write_text("")
        adapter = LLMServingSimAdapter(str(tmp_path))
        assert os.path.isabs(adapter.llmservingsim_dir)

    def test_init_rejects_missing_main_py(self, tmp_path):
        """Constructor should raise ValueError if main.py not found."""
        with pytest.raises(ValueError, match="Invalid LLMServingSim directory"):
            LLMServingSimAdapter(str(tmp_path))
