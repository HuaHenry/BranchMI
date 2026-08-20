from pathlib import Path

from branchmi_pilot.config import load_config


def test_smoke_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "smoke.yaml")
    assert cfg.dataset.kind == "jsonl"
    assert cfg.generation.candidate_top_k == 2
    assert cfg.checkpoints.max_per_problem == 2

