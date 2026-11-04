import json
from datetime import datetime

from scripts import run_batch
from scripts.run_batch import IST, compare, metrics, normalize, report, seconds_until

OLD_ROW = {"instance_id": "a__b-1", "repo": "a", "run_id": "r", "duration_s": 100, "note": "", "status": "awaiting_review",
           "verified": True, "resolved": False, "oracle_ran": True, "gold_file_found": True, "repro_valid": False,
           "attempts": 1, "attempts_applied": 1, "tokens_in": 10, "tokens_out": 5, "failure_reason": ""}


def test_pre_h1_row_is_rescored():
    r = normalize(OLD_ROW)
    assert r["verified"] is False and r["regression_only_pass"] is True


def test_pre_h1_row_with_a_repro_stays_verified():
    assert normalize({**OLD_ROW, "repro_valid": True})["verified"] is True


def test_new_row_is_left_alone():
    row = {**OLD_ROW, "verified": False, "regression_only_pass": False}
    assert normalize(row) == row


def test_report_tolerates_rows_missing_newer_fields():
    assert "Lint repairs | 0" in report([{"instance_id": "x__y-1"}, OLD_ROW])


def test_excluded_instance_is_out_of_the_resolved_denominator():
    m = metrics([{**OLD_ROW, "instance_id": "psf__requests-2317", "resolved": True}, OLD_ROW])
    assert (m["resolved"], m["fair_n"], m["n"]) == (0, 1, 2)


def test_false_positive_includes_regression_only_pass():
    assert metrics([OLD_ROW])["false_positives"] == 1


def test_false_negative():
    row = {**OLD_ROW, "status": "failed", "verified": False, "regression_only_pass": False, "resolved": True}
    assert metrics([row])["false_negatives"] == 1


def test_compare_lists_each_tag(tmp_path):
    for tag in ("qwen-v1", "qwen-v2-full"):
        (tmp_path / tag).mkdir()
        (tmp_path / tag / "batch_results.json").write_text(json.dumps([OLD_ROW]))
    text = compare(tmp_path)
    assert "| qwen-v1 | 1 |" in text and "| qwen-v2-full | 1 |" in text and "predates" in text


def test_seconds_until_rolls_to_next_day():
    now = datetime(2026, 11, 5, 1, 0, tzinfo=IST)
    assert seconds_until("04:15", now) == 3.25 * 3600
    assert seconds_until("00:30", now) == 23.5 * 3600


def test_tag_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "OUT_DIR", tmp_path)
    assert run_batch.paths("t")[0] == tmp_path / "t" / "batch_results.json"
    assert run_batch.paths(None)[1] == tmp_path / "batch_results.md"
