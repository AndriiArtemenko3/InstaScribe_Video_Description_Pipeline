"""Plain factory helpers for benchmark tests (no pytest fixtures on purpose).

Each helper builds a valid object with overridable defaults, so a test only
spells out the fields it is actually about.
"""

from __future__ import annotations

from instadescribe_benchmarks.id_event_light_v0.schema import (
    BenchmarkItem,
    Difficulty,
    Prediction,
)


def item(
    item_id: str,
    *,
    difficulty: Difficulty = Difficulty.POSITIVE,
    start: float | None = 2.0,
    end: float | None = 5.0,
    query: str = "Does a person enter the vehicle?",
    notes: str | None = None,
) -> BenchmarkItem:
    if difficulty is not Difficulty.POSITIVE:
        start = None
        end = None
    return BenchmarkItem(
        item_id=item_id,
        video=f"videos/{item_id}.mp4",
        query=query,
        label=difficulty is Difficulty.POSITIVE,
        start_time=start,
        end_time=end,
        difficulty=difficulty,
        notes=notes,
    )


def prediction(
    item_id: str,
    *,
    detected: bool = True,
    start: float | None = 2.0,
    end: float | None = 5.0,
    confidence: float | None = 0.9,
) -> Prediction:
    return Prediction(
        item_id=item_id,
        event_detected=detected,
        start_time=start,
        end_time=end,
        confidence=confidence,
    )
