"""Tamper-evident XR02 observation journal and content-addressed embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from spatial_mapping_phase2.xr02_local_domain import (
    EmbeddingReference,
    LocalTrackObservation,
)

_GENESIS = "0" * 64


class XR02JournalError(RuntimeError):
    """Raised when append-only evidence cannot be written or verified."""


@dataclass(frozen=True, slots=True)
class JournalVerification:
    records: int
    final_sha256: str


class EmbeddingStore:
    """Store normalized float32 vectors once, addressed by exact NPY bytes."""

    def __init__(self, root: Path, model_id: str) -> None:
        self.root = root
        self.model_id = model_id

    def put(self, vector: NDArray[np.floating[Any]]) -> EmbeddingReference:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise XR02JournalError("embedding must contain finite values")
        norm = float(np.linalg.norm(values))
        if norm <= 0:
            raise XR02JournalError("embedding norm must be positive")
        normalized = np.ascontiguousarray(values / norm, dtype=np.float32)

        temporary_root = self.root / ".tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = temporary_root / f"{uuid4().hex}.npy"
        with temporary.open("wb") as handle:
            np.save(handle, normalized, allow_pickle=False)
        payload = temporary.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path("embeddings") / digest[:2] / f"{digest}.npy"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                temporary.unlink(missing_ok=True)
                raise XR02JournalError("content-addressed embedding collision")
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, destination)
        try:
            temporary_root.rmdir()
        except OSError:
            pass
        return EmbeddingReference(
            sha256=digest,
            model_id=self.model_id,
            dimension=int(normalized.size),
            relative_path=relative.as_posix(),
        )

    def load(self, reference: EmbeddingReference) -> NDArray[np.float32]:
        path = self.root / reference.relative_path
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference.sha256:
            raise XR02JournalError("embedding identity changed")
        with path.open("rb") as handle:
            vector = np.asarray(np.load(handle, allow_pickle=False), dtype=np.float32)
        if vector.shape != (reference.dimension,) or not np.all(np.isfinite(vector)):
            raise XR02JournalError("stored embedding violates its reference")
        return vector


class ObservationJournal:
    """Append canonical JSONL records linked by a SHA-256 chain."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists() and path.stat().st_size:
            verification = verify_journal(path)
            self._sequence = verification.records
            self._previous_sha256 = verification.final_sha256
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._sequence = 0
            self._previous_sha256 = _GENESIS

    def append(self, observation: LocalTrackObservation) -> str:
        return self.append_batch((observation,))[-1]

    def append_batch(self, observations: tuple[LocalTrackObservation, ...]) -> tuple[str, ...]:
        """Append one local tick atomically enough for bounded live evidence.

        Records retain the exact per-row SHA-256 chain used by ``append``.  The
        live path opens, flushes and synchronizes once per non-empty tick instead
        of once per person, removing person-count-dependent fsync amplification.
        """

        if not observations:
            return ()
        encoded_records: list[str] = []
        digests: list[str] = []
        sequence = self._sequence
        previous = self._previous_sha256
        for observation in observations:
            core: dict[str, object] = {
                "sequence": sequence,
                "previous_sha256": previous,
                "payload": observation.as_dict(),
            }
            record_sha256 = _hash_mapping(core)
            encoded_records.append(_canonical_json({**core, "record_sha256": record_sha256}))
            digests.append(record_sha256)
            sequence += 1
            previous = record_sha256
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for encoded in encoded_records:
                handle.write(encoded)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence = sequence
        self._previous_sha256 = previous
        return tuple(digests)


def verify_journal(path: Path) -> JournalVerification:
    previous = _GENESIS
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise XR02JournalError(f"journal line {line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise XR02JournalError(f"journal line {line_number} is not an object")
            if record.get("sequence") != records:
                raise XR02JournalError(f"journal line {line_number} sequence changed")
            if record.get("previous_sha256") != previous:
                raise XR02JournalError(f"journal line {line_number} chain changed")
            claimed = record.get("record_sha256")
            core = {key: value for key, value in record.items() if key != "record_sha256"}
            actual = _hash_mapping(core)
            if claimed != actual:
                raise XR02JournalError(f"journal line {line_number} content changed")
            previous = actual
            records += 1
    return JournalVerification(records=records, final_sha256=previous)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_mapping(value: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()
