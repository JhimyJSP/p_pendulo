from pendulum_project.training import _validation_decisions


def test_checkpoint_tracks_minimum_even_below_patience_threshold():
    save_checkpoint, reset_patience = _validation_decisions(
        loss=0.95, best_loss=1.0, patience_reference=1.0, min_delta=0.1)
    assert save_checkpoint
    assert not reset_patience
