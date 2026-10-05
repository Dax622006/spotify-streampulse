# spotify-streampulse/tests/test_whatif.py
"""
Unit tests for What-If scenario simulator and immutability guard per Phase 15.
"""
from __future__ import annotations
import sqlite3
import pytest

from streampulse.config import SOURCE_FILE, WAREHOUSE_FILE
from streampulse.ingestion.loader import sha256_of
from streampulse.ml.inference import InputSpec, predict_scenario, validate_whatif_input
from streampulse.warehouse.builder import warehouse_content_hash


@pytest.fixture
def input_spec():
    return InputSpec(
        streams_min=1,
        streams_max=20,
        duration_min=2,
        duration_max=100,
        devices=["Mobile", "Desktop", "Web", "Tablet"],
        genres=["Pop", "Rock", "Hip-Hop"],
        countries=["USA", "UK", "Canada"],
    )


def test_whatif_input_validation(input_spec):
    # Valid input
    ok, errs, warns = validate_whatif_input(10, 30, "Mobile", "Pop", "USA", input_spec)
    assert ok is True
    assert len(errs) == 0

    # Invalid streams
    ok, errs, warns = validate_whatif_input(50, 30, "Mobile", "Pop", "USA", input_spec)
    assert ok is False
    assert any("Streams" in e for e in errs)

    # Invalid category
    ok, errs, warns = validate_whatif_input(10, 30, "InvalidDevice", "Pop", "USA", input_spec)
    assert ok is False
    assert any("device" in e.lower() for e in errs)

    # Ratio warning
    ok, errs, warns = validate_whatif_input(10, 80, "Mobile", "Pop", "USA", input_spec)
    assert ok is True
    assert len(warns) > 0  # 80/10 = 8.0 > 5.0


def test_whatif_predictions():
    pred = predict_scenario(
        streams=10,
        duration=30,
        device="Mobile",
        genre="Pop",
        country="USA",
    )
    assert pred.slr_revenue >= 0.0
    assert pred.mlr_revenue >= 0.0
    assert pred.dt_subscription in ["Free", "Premium"]
    assert pred.nb_subscription in ["Free", "Premium"]


def test_whatif_immutability():
    """R-WI-02, AC-WI-3: Running inferences must NOT mutate warehouse or source."""
    initial_source_sha = sha256_of(SOURCE_FILE)
    conn = sqlite3.connect(f"file:{WAREHOUSE_FILE}?mode=ro", uri=True)
    initial_warehouse_hash = warehouse_content_hash(conn)
    conn.close()

    # Run 25 consecutive inferences
    for i in range(1, 26):
        predict_scenario(
            streams=(i % 15) + 1,
            duration=((i % 15) + 1) * 3,
            device="Mobile",
            genre="Pop",
            country="USA",
        )

    # Verify hashes unchanged
    assert sha256_of(SOURCE_FILE) == initial_source_sha
    conn = sqlite3.connect(f"file:{WAREHOUSE_FILE}?mode=ro", uri=True)
    post_warehouse_hash = warehouse_content_hash(conn)
    conn.close()
    assert post_warehouse_hash == initial_warehouse_hash
