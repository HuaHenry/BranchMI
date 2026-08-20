import json

from branchmi_pilot.analysis import analyze_run
from branchmi_pilot.config import AnalysisConfig, ExperimentConfig, save_config


def test_analysis_writes_metrics_report_and_figures(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    cfg = ExperimentConfig(
        run_name="synthetic",
        output_dir=str(tmp_path),
        analysis=AnalysisConfig(bootstrap_samples=10),
    )
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "resolved_config.yaml")

    rows = []
    for index in range(8):
        critical = index >= 4
        rows.append(
            {
                "problem_id": f"p-{index}",
                "checkpoint_index": 0,
                "generated_position": 32,
                "oracle_answer_change": critical,
                "oracle_correctness_change": critical,
                "probe_oracle_branch_agreement": 0.75,
                "entropy_nats": float(8 - index),
                "varentropy_nats2": float(index),
                "lookahead_js_mean_nats": float(index) / 8,
                "branchmi_weighted_nats": float(index),
                "branchmi_uniform_nats": float(index),
            }
        )
    with (run_dir / "checkpoints.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary = analyze_run(run_dir)

    assert summary["scores"]["branchmi_weighted_nats"]["auroc"] == 1.0
    assert summary["scores"]["entropy_nats"]["auroc"] == 0.0
    assert (run_dir / "report.md").exists()
    assert (run_dir / "checkpoint_metrics.csv").exists()
    assert (run_dir / "figures" / "roc_curves.png").exists()
