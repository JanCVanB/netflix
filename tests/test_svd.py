import numpy as np

from algorithms import svd


def help_get_simple_ratings():
    ratings = ((1, 0, 2),
               (0, 3, 0),
               (4, 0, 5),
               (0, 1, 0))
    return np.array(ratings)


def test_run_returns_result_of_train_on_ratings(monkeypatch):
    from random import random
    unique_value = random()
    dummy_train = lambda ratings: unique_value
    monkeypatch.setattr(svd, 'train', dummy_train)

    simple_ratings = help_get_simple_ratings()
    expected_result = svd.train(simple_ratings)
    actual_result = svd.run(simple_ratings)
    assert actual_result == expected_result
