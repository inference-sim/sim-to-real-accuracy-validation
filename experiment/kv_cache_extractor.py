"""Extract KV cache block counts from vLLM ground-truth logs.

Two extraction functions:

* ``extract_total_kv_blocks`` — parse *vllm.log* for the GPU KV cache size
  line and convert token count to block count (tokens / 16).
* ``extract_cpu_kv_blocks`` — stream *kv_events.jsonl* line-by-line, tracking
  CPU-resident block counts via ``TransferCompleted`` events and return the
  **peak** concurrent count.

.. note::
   ``CacheStoreCommitted`` and ``BlockStored`` events describe the *same*
   blocks at different lifecycle stages as ``TransferCompleted``.  Only
   ``TransferCompleted`` is counted to avoid double-counting.
"""

from __future__ import annotations

import json
import re

_KV_CACHE_RE = re.compile(r"GPU KV cache size:\s+([\d,]+)\s+tokens")

BLOCK_SIZE = 16


def extract_total_kv_blocks(vllm_log_path: str) -> int:
    """Return the total GPU KV cache block count from a vllm.log file.

    Searches for a line matching::

        INFO vllm.v1.core.kv_cache_utils: GPU KV cache size: 119,408 tokens

    Strips commas from the token count, converts to int, and divides by
    ``BLOCK_SIZE`` (16) using integer division.

    Parameters
    ----------
    vllm_log_path:
        Filesystem path to the ``vllm.log`` file.

    Returns
    -------
    int
        Number of KV cache blocks (tokens // 16).

    Raises
    ------
    ValueError
        If no matching line is found in the log file.
    """
    with open(vllm_log_path, "r") as fh:
        for line in fh:
            match = _KV_CACHE_RE.search(line)
            if match:
                tokens = int(match.group(1).replace(",", ""))
                return tokens // BLOCK_SIZE

    raise ValueError(
        f"No 'GPU KV cache size' line found in {vllm_log_path}"
    )


def extract_cpu_kv_blocks(kv_events_path: str) -> int:
    """Return the peak CPU-resident KV cache block count from kv_events.jsonl.

    Each line in the file is a JSON array with the structure::

        [timestamp, [event1, event2, ...], flags, metadata]

    Only ``TransferCompleted`` events are counted because
    ``CacheStoreCommitted`` and ``BlockStored`` describe the *same* blocks
    at earlier lifecycle stages (committed intent vs. actual transfer).

    * ``"TransferCompleted"`` — ``[type, transfer_id, req_id, src, dst,
      block_count, success, seq]``.  GPU→CPU increments; CPU→GPU decrements.

    The file is processed **line-by-line** to avoid loading hundreds of MB
    into memory at once.

    Parameters
    ----------
    kv_events_path:
        Filesystem path to the ``kv_events.jsonl`` file.

    Returns
    -------
    int
        Peak number of blocks concurrently resident on CPU.
    """
    current_cpu_blocks = 0
    peak_cpu_blocks = 0

    with open(kv_events_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # record: [timestamp, [event1, event2, ...], flags, metadata]
            if not isinstance(record, list) or len(record) < 2:
                continue
            events = record[1]
            if not isinstance(events, list):
                continue

            for event in events:
                if not isinstance(event, list) or not event:
                    continue
                event_type = event[0]

                if event_type == "TransferCompleted":
                    # event: ["TransferCompleted", transfer_id, req_id,
                    #          src_tier, dst_tier, block_count, success, seq]
                    if len(event) < 7:
                        continue
                    success = event[6]
                    if not success:
                        continue
                    src_tier = event[3]
                    dst_tier = event[4]
                    block_count = event[5]

                    if src_tier == "GPU" and dst_tier == "CPU":
                        current_cpu_blocks += block_count
                    elif src_tier == "CPU" and dst_tier == "GPU":
                        current_cpu_blocks = max(0, current_cpu_blocks - block_count)

            # Update peak after processing all events in this line.
            if current_cpu_blocks > peak_cpu_blocks:
                peak_cpu_blocks = current_cpu_blocks

    return peak_cpu_blocks
