def test_run_returns_numpy_array():
    import numpy as np
    from algorithms.svd import run
    output_array = run()
    assert isinstance(output_array, np.ndarray)


def test_run_can_accept_numpy_array_as_only_argument():
    import numpy as np
    from algorithms.svd import run
    input_array = np.array([])
    run(input_array)
