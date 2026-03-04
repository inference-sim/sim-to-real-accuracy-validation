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

    def test_empty_file_returns_zero(self, tmp_path: str) -> None:
        """An empty kv_events.jsonl should return 0."""
        path = _write_kv_events(str(tmp_path), [])
        assert extract_cpu_kv_blocks(path) == 0

    def test_no_cpu_events_returns_zero(self, tmp_path: str) -> None:
        """Events that don't touch CPU should yield 0."""
        records = [
            [1.0, [["SomeOtherEvent", "a", "b", "c"]], {}, {}],
            [2.0, [["BlockStored", "x", "y", "z", "a", "b", "GPU"]], {}, {}],
        ]
        path = _write_kv_events(str(tmp_path), records)
        assert extract_cpu_kv_blocks(path) == 0

    def test_block_stored_cpu(self, tmp_path: str) -> None:
        """BlockStored events with 'CPU' at position [6] increment the count."""
        records = [
            # Three BlockStored CPU events in one line
            [
                1.0,
                [
                    ["BlockStored", "a", "b", "c", "d", "e", "CPU"],
                    ["BlockStored", "a", "b", "c", "d", "e", "CPU"],
                    ["BlockStored", "a", "b", "c", "d", "e", "GPU"],  # not CPU
                ],
                {},
                {},
            ],
            # One more BlockStored CPU
            [
                2.0,
                [["BlockStored", "a", "b", "c", "d", "e", "CPU"]],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        # Peak after line 1: 2, after line 2: 3  => peak = 3
        assert extract_cpu_kv_blocks(path) == 3

    def test_transfer_completed_gpu_to_cpu(self, tmp_path: str) -> None:
        """TransferCompleted GPU->CPU increments by block_count."""
        records = [
            [
                1.0,
                [
                    # ["TransferCompleted", transfer_id, req_id, src, dst, block_count, success, seq]
                    ["TransferCompleted", 1, "req1", "GPU", "CPU", 10, True, 0],
                ],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        assert extract_cpu_kv_blocks(path) == 10

    def test_transfer_completed_cpu_to_gpu(self, tmp_path: str) -> None:
        """TransferCompleted CPU->GPU decrements by block_count."""
        records = [
            # First: put 10 blocks on CPU
            [
                1.0,
                [["TransferCompleted", 1, "req1", "GPU", "CPU", 10, True, 0]],
                {},
                {},
            ],
            # Then: move 4 blocks back to GPU
            [
                2.0,
                [["TransferCompleted", 2, "req2", "CPU", "GPU", 4, True, 1]],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        # Peak was 10 (after line 1); drops to 6 after line 2 but peak is still 10.
        assert extract_cpu_kv_blocks(path) == 10

    def test_cache_store_committed_cpu(self, tmp_path: str) -> None:
        """CacheStoreCommitted with tier='CPU' increments by block_count."""
        records = [
            [
                1.0,
                [
                    # ["CacheStoreCommitted", req_id, tier, block_count, seq]
                    ["CacheStoreCommitted", "req1", "CPU", 5, 0],
                ],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        assert extract_cpu_kv_blocks(path) == 5

    def test_cache_store_committed_gpu_ignored(self, tmp_path: str) -> None:
        """CacheStoreCommitted with tier='GPU' should not affect CPU count."""
        records = [
            [
                1.0,
                [["CacheStoreCommitted", "req1", "GPU", 5, 0]],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        assert extract_cpu_kv_blocks(path) == 0

    def test_peak_tracking_across_lines(self, tmp_path: str) -> None:
        """Verify peak is tracked correctly when count rises then falls."""
        records = [
            # +10
            [1.0, [["TransferCompleted", 1, "r1", "GPU", "CPU", 10, True, 0]], {}, {}],
            # +5 -> 15
            [2.0, [["CacheStoreCommitted", "r2", "CPU", 5, 1]], {}, {}],
            # -8 -> 7
            [3.0, [["TransferCompleted", 2, "r3", "CPU", "GPU", 8, True, 2]], {}, {}],
            # +2 -> 9
            [4.0, [["CacheStoreCommitted", "r4", "CPU", 2, 3]], {}, {}],
        ]
        path = _write_kv_events(str(tmp_path), records)
        # Peak is 15 (after line 2)
        assert extract_cpu_kv_blocks(path) == 15

    def test_mixed_events_in_single_line(self, tmp_path: str) -> None:
        """Multiple event types in a single JSONL line are all processed."""
        records = [
            [
                1.0,
                [
                    ["BlockStored", "a", "b", "c", "d", "e", "CPU"],  # +1
                    ["TransferCompleted", 1, "r1", "GPU", "CPU", 5, True, 0],  # +5
                    ["CacheStoreCommitted", "r2", "CPU", 3, 1],  # +3
                ],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        # Total: 1 + 5 + 3 = 9
        assert extract_cpu_kv_blocks(path) == 9

    def test_block_stored_short_event_ignored(self, tmp_path: str) -> None:
        """A BlockStored event with fewer than 7 elements should be ignored."""
        records = [
            [
                1.0,
                [
                    ["BlockStored", "a", "b"],  # too short, no position [6]
                ],
                {},
                {},
            ],
        ]
        path = _write_kv_events(str(tmp_path), records)
        assert extract_cpu_kv_blocks(path) == 0

    def test_blank_lines_are_skipped(self, tmp_path: str) -> None:
        """Blank lines in the JSONL file should be silently skipped."""
        path = os.path.join(str(tmp_path), "kv_events.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps([1.0, [["CacheStoreCommitted", "r1", "CPU", 7, 0]], {}, {}]) + "\n")
            fh.write("\n")  # blank line
            fh.write("   \n")  # whitespace-only line
            fh.write(json.dumps([2.0, [["CacheStoreCommitted", "r2", "CPU", 3, 1]], {}, {}]) + "\n")
        assert extract_cpu_kv_blocks(path) == 10
