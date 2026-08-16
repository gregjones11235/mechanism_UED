from pathlib import Path


PPO_SOURCE = Path(__file__).parents[1] / "ppo_tr.py"


def test_callback_free_path_collects_metrics_without_removing_baseline():
    source = PPO_SOURCE.read_text(encoding="utf-8")
    assert 'config.get("runtime", {}).get("host_callback_free", False)' in source
    assert "if not host_callback_free:" in source
    assert '"train_metrics": scan_train_metrics' in source
    assert "jax.debug.callback(_log_callback, metrics_to_log, current_step)" in source


def test_callback_free_path_keeps_all_six_logged_fields():
    source = PPO_SOURCE.read_text(encoding="utf-8")
    for key in (
        "train/total_loss",
        "train/value_loss",
        "train/actor_loss",
        "train/entropy",
        "train/grad_norm_mean",
        "train/grad_norm_max",
    ):
        assert key in source
