"""Tamper-evident, append-only evidence journal for XR02 global decisions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from spatial_mapping_phase2.xr02_global_domain import AssociationTickResult

_GENESIS = "0" * 64


class GlobalAssociationJournalError(RuntimeError):
    """Raised when global association evidence cannot be written or verified."""


@dataclass(frozen=True, slots=True)
class GlobalJournalVerification:
    records: int
    final_sha256: str
    signature_sha256: str


class GlobalAssociationJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists() and path.stat().st_size:
            verified = verify_global_journal(path)
            self._sequence = verified.records
            self._previous_sha256 = verified.final_sha256
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._sequence = 0
            self._previous_sha256 = _GENESIS

    def append(self, result: AssociationTickResult) -> str:
        return self.append_batch((result,))[-1]

    def append_batch(self, results: tuple[AssociationTickResult, ...]) -> tuple[str, ...]:
        """Persist a bounded association micro-batch with one flush/fsync."""

        if not results:
            return ()
        sequence = self._sequence
        previous = self._previous_sha256
        encoded_records: list[str] = []
        digests: list[str] = []
        for result in results:
            core: dict[str, object] = {
                "sequence": sequence,
                "previous_sha256": previous,
                "payload": result.as_dict(),
                "decision_signature_sha256": result.signature_sha256,
            }
            record_sha256 = _sha256(core)
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


def verify_global_journal(path: Path) -> GlobalJournalVerification:
    previous = _GENESIS
    signatures: list[str] = []
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise GlobalAssociationJournalError(
                    f"global journal line {line_number} is invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise GlobalAssociationJournalError("global journal record is not an object")
            if record.get("sequence") != records or record.get("previous_sha256") != previous:
                raise GlobalAssociationJournalError("global journal sequence or chain changed")
            claimed = record.get("record_sha256")
            core = {key: value for key, value in record.items() if key != "record_sha256"}
            actual = _sha256(core)
            if claimed != actual:
                raise GlobalAssociationJournalError("global journal content changed")
            signature = record.get("decision_signature_sha256")
            if not isinstance(signature, str):
                raise GlobalAssociationJournalError("global journal decision signature missing")
            signatures.append(signature)
            previous = actual
            records += 1
    return GlobalJournalVerification(records, previous, _sha256(signatures))


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()
