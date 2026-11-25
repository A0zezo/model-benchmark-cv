from bench import run, MODELS


def test_all_models_scored():
    r = run()
    assert set(r) == set(MODELS) and all(len(v) == 5 for v in r.values())


def test_beats_majority_baseline():
    # majority class is about 63% of this dataset
    assert all(v.mean() > 0.63 for v in run().values())


def test_deterministic():
    a, b = run(), run()
    assert all((a[k] == b[k]).all() for k in a)
