"""Tests for experiment.kv_cache_extractor."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from experiment.kv_cache_extractor import (
    BLOCK_SIZE,
    extract_cpu_kv_blocks,
    extract_total_kv_blocks,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic fixture files into a temp directory
# ---------------------------------------------------------------------------


def _write_vllm_log(tmpdir: str, lines: list[str]) -> str:
    """Write *lines* to a ``vllm.log`` file inside *tmpdir* and return its path."""
    path = os.path.join(tmpdir, "vllm.log")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _write_kv_events(tmpdir: str, records: list) -> str:
    """Write JSONL records to ``kv_events.jsonl`` inside *tmpdir*.

    Each element of *records* should be a complete JSON-serialisable record
    (i.e. ``[timestamp, [events...], flags, metadata]``).
    """
    path = os.path.join(tmpdir, "kv_events.jsonl")
    with open(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests for extract_total_kv_blocks
# ---------------------------------------------------------------------------


class TestExtractTotalKvBlocks:
    """Unit tests for ``extract_total_kv_blocks``."""

    def test_llama2_7b(self, tmp_path: str) -> None:
        """Llama-2-7B: 119,408 tokens -> 7,463 blocks."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO 01-15 10:00:00 some_module.py: Starting up...",
                "INFO 01-15 10:00:01 vllm.v1.core.kv_cache_utils: GPU KV cache size: 119,408 tokens",
                "INFO 01-15 10:00:02 some_module.py: Ready.",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 119_408 // BLOCK_SIZE  # 7463

    def test_llama2_70b(self, tmp_path: str) -> None:
        """Llama-2-70B: 501,712 tokens -> 31,357 blocks."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 501,712 tokens",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 501_712 // BLOCK_SIZE  # 31357

    def test_mixtral_8x7b(self, tmp_path: str) -> None:
        """Mixtral-8x7B: 440,176 tokens -> 27,511 blocks."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 440,176 tokens",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 440_176 // BLOCK_SIZE  # 27511

    def test_codellama_34b(self, tmp_path: str) -> None:
        """CodeLlama-34B: 425,648 tokens -> 26,603 blocks."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 425,648 tokens",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 425_648 // BLOCK_SIZE  # 26603

    def test_no_commas_in_token_count(self, tmp_path: str) -> None:
        """Token count without commas should still parse."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 1600 tokens",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 1600 // BLOCK_SIZE  # 100

    def test_line_found_among_many(self, tmp_path: str) -> None:
        """The relevant line is buried among many unrelated lines."""
        lines = [f"INFO line {i}" for i in range(100)]
        lines.insert(
            42,
            "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 3,200 tokens",
        )
        log_path = _write_vllm_log(str(tmp_path), lines)
        assert extract_total_kv_blocks(log_path) == 3_200 // BLOCK_SIZE  # 200

    def test_first_match_wins(self, tmp_path: str) -> None:
        """If the line appears more than once, the first occurrence is used."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 1,600 tokens",
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 3,200 tokens",
            ],
        )
        assert extract_total_kv_blocks(log_path) == 1_600 // BLOCK_SIZE  # 100

    def test_missing_log_line_raises(self, tmp_path: str) -> None:
        """A log with no matching line should raise ValueError."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO some_module: nothing relevant here",
                "WARNING: something else",
            ],
        )
        with pytest.raises(ValueError, match="No 'GPU KV cache size' line found"):
            extract_total_kv_blocks(log_path)

    def test_empty_log_raises(self, tmp_path: str) -> None:
        """An empty log file should raise ValueError."""
        log_path = _write_vllm_log(str(tmp_path), [])
        with pytest.raises(ValueError, match="No 'GPU KV cache size' line found"):
            extract_total_kv_blocks(log_path)


# ---------------------------------------------------------------------------
# Tests for extract_cpu_kv_blocks
# ---------------------------------------------------------------------------


class TestExtractCpuKvBlocks:
    """Unit tests for ``extract_cpu_kv_blocks``."""

    def test_no_cpu_offloading_returns_zero(self, tmp_path: str) -> None:
        """Log without cpu_bytes_to_use should return 0."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 425,648 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 38.97 GiB",
            ],
        )
        assert extract_cpu_kv_blocks(log_path) == 0

    def test_codellama_34b_8gb_cpu(self, tmp_path: str) -> None:
        """CodeLlama-34B with 8 GB CPU offloading."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 425,648 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 38.97 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}",
            ],
        )
        # 38.97 GiB = 41,843,040,256 bytes
        # bytes_per_token = 41,843,040,256 / 425,648 ≈ 98,279 bytes/token
        # bytes_per_block = 98,279 * 16 ≈ 1,572,464 bytes/block
        # cpu_blocks = 8,589,934,592 / 1,572,464 ≈ 5,462 blocks
        result = extract_cpu_kv_blocks(log_path)
        assert 5400 < result < 5500  # Allow some tolerance for float rounding

    def test_mixtral_8x7b_8gb_cpu(self, tmp_path: str) -> None:
        """Mixtral-8x7B with 8 GB CPU offloading."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 440,176 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 39.18 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}",
            ],
        )
        # 39.18 GiB = 42,068,525,056 bytes
        # bytes_per_token = 42,068,525,056 / 440,176 ≈ 95,566 bytes/token
        # bytes_per_block = 95,566 * 16 ≈ 1,529,056 bytes/block
        # cpu_blocks = 8,589,934,592 / 1,529,056 ≈ 5,617 blocks
        result = extract_cpu_kv_blocks(log_path)
        assert 5550 < result < 5650

    def test_llama2_7b_4gb_cpu(self, tmp_path: str) -> None:
        """Llama-2-7B with 4 GB CPU offloading."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 119,408 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 12.50 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 4294967296.0}",
            ],
        )
        # 12.50 GiB = 13,421,772,800 bytes
        # bytes_per_token = 13,421,772,800 / 119,408 ≈ 112,413 bytes/token
        # bytes_per_block = 112,413 * 16 ≈ 1,798,608 bytes/block
        # cpu_blocks = 4,294,967,296 / 1,798,608 ≈ 2,387 blocks
        result = extract_cpu_kv_blocks(log_path)
        assert 2350 < result < 2450

    def test_cpu_bytes_without_quotes(self, tmp_path: str) -> None:
        """cpu_bytes_to_use without quotes around key should parse."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 100,000 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 10.0 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={cpu_bytes_to_use: 2147483648.0}",
            ],
        )
        result = extract_cpu_kv_blocks(log_path)
        # Should parse successfully
        assert result > 0

    def test_missing_gpu_cache_size_raises(self, tmp_path: str) -> None:
        """cpu_bytes_to_use without GPU cache size should raise."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 10.0 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}",
            ],
        )
        with pytest.raises(ValueError, match="no 'GPU KV cache size' line"):
            extract_cpu_kv_blocks(log_path)

    def test_missing_available_memory_raises(self, tmp_path: str) -> None:
        """cpu_bytes_to_use without available memory should raise."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 100,000 tokens",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}",
            ],
        )
        with pytest.raises(ValueError, match="no 'Available KV cache memory' line"):
            extract_cpu_kv_blocks(log_path)

    def test_first_occurrence_used(self, tmp_path: str) -> None:
        """If cpu_bytes_to_use appears multiple times, first is used."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 100,000 tokens",
                "INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 10.0 GiB",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 4294967296.0}",
                "INFO vllm.v1.worker.gpu_worker: kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}",
            ],
        )
        # Should use first value (4 GB)
        result = extract_cpu_kv_blocks(log_path)
        # 10 GiB = 10,737,418,240 bytes
        # bytes_per_token = 10,737,418,240 / 100,000 = 107,374 bytes/token
        # bytes_per_block = 107,374 * 16 = 1,717,984 bytes/block
        # cpu_blocks = 4,294,967,296 / 1,717,984 ≈ 2,500 blocks
        assert 2450 < result < 2550

    def test_real_vllm_log_format(self, tmp_path: str) -> None:
        """Test with actual vLLM v0.15.1 log format."""
        log_path = _write_vllm_log(
            str(tmp_path),
            [
                "2026-02-18 15:34:47,493 INFO vllm.v1.worker.gpu_worker: Available KV cache memory: 38.97 GiB",
                "2026-02-18 15:34:47,640 INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 425,648 tokens",
                "2026-02-18 15:34:47,642 INFO vllm.v1.worker.gpu_worker: GPUWorker.initialize_from_config called with kv_transfer_config=KVTransferConfig(kv_connector='OffloadingConnector', engine_id='xyz', kv_buffer_device='cuda', kv_buffer_size=1000000000.0, kv_role='kv_both', kv_rank=None, kv_parallel_size=1, kv_ip='127.0.0.1', kv_port=14579, kv_connector_extra_config={'cpu_bytes_to_use': 8589934592.0}, kv_connector_module_path=None, enable_permute_local_kv=False, kv_load_failure_policy='recompute')",
            ],
        )
        result = extract_cpu_kv_blocks(log_path)
        # Same calculation as test_codellama_34b_8gb_cpu
        assert 5400 < result < 5500
