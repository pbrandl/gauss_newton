import numpy as np
import pytest

from gauss_newton import GaussNewton, Polynomial, finite_difference_jacobian, rmse

METHODS = ["forward", "backward", "central"]


def exponential_decay(params, x):
    amplitude, rate, offset = params
    return amplitude * np.exp(-rate * x) + offset


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_of_polynomial_is_vandermonde_matrix(method):
    x = np.linspace(-1.5, 1.5, 11)
    params = np.array([0.5, -1.0, 2.0, 0.3])
    jacobian = finite_difference_jacobian(Polynomial(), params, x, method=method)
    expected = np.vander(x, len(params), increasing=True)
    np.testing.assert_allclose(jacobian, expected, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_recovers_polynomial_from_exact_data(method):
    model = Polynomial()
    true_params = np.array([1.0, -2.0, 0.5, 0.25])
    x = np.linspace(-1.5, 1.5, 20)

    result = GaussNewton(method=method).fit(model, x, model(true_params, x), np.zeros(4))

    assert result.converged
    np.testing.assert_allclose(result.params, true_params, atol=1e-6)
    assert result.error == pytest.approx(0.0, abs=1e-8)


def test_noisy_polynomial_fit_matches_linear_least_squares():
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.5, 1.5, size=30)
    y = 1.0 - x + 2.0 * x**3 + rng.normal(0.0, 0.5, size=x.size)

    result = GaussNewton().fit(Polynomial(), x, y, np.zeros(4))

    np.testing.assert_allclose(result.params, np.polynomial.polynomial.polyfit(x, y, 3), atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_fits_nonlinear_model(method):
    true_params = np.array([2.0, 1.5, 0.5])
    x = np.linspace(0.0, 4.0, 40)
    y = exponential_decay(true_params, x)

    result = GaussNewton(method=method).fit(exponential_decay, x, y, [1.0, 1.0, 0.0])

    assert result.converged
    np.testing.assert_allclose(result.params, true_params, atol=1e-5)


def test_history_is_monotonic_for_nonlinear_model():
    x = np.linspace(0.0, 4.0, 40)
    y = exponential_decay([2.0, 1.5, 0.5], x)

    result = GaussNewton().fit(exponential_decay, x, y, [1.0, 1.0, 0.0])

    assert len(result.history) == result.n_iterations + 1
    assert np.all(np.diff(result.history) <= 1e-12)


def test_params_history_matches_error_history():
    x = np.linspace(0.0, 4.0, 40)
    y = exponential_decay([2.0, 1.5, 0.5], x)
    initial_params = [1.0, 1.0, 0.0]

    result = GaussNewton().fit(exponential_decay, x, y, initial_params)

    assert len(result.params_history) == len(result.history)
    np.testing.assert_array_equal(result.params_history[0], initial_params)
    np.testing.assert_array_equal(result.params_history[-1], result.params)
    for params, error in zip(result.params_history, result.history, strict=True):
        assert rmse(y, exponential_decay(params, x)) == pytest.approx(error)


def test_does_not_modify_initial_params_and_preserves_shape():
    model = Polynomial()
    x = np.linspace(-1, 1, 10).reshape(1, -1)
    initial_params = np.zeros((3, 1))

    result = GaussNewton().fit(model, x, model([[1.0], [2.0], [3.0]], x), initial_params)

    np.testing.assert_array_equal(initial_params, np.zeros((3, 1)))
    assert result.params.shape == (3, 1)
    np.testing.assert_allclose(result.params.ravel(), [1.0, 2.0, 3.0], atol=1e-6)


def test_reports_non_convergence_when_iterations_run_out():
    x = np.linspace(0.0, 4.0, 40)
    y = exponential_decay([2.0, 1.5, 0.5], x)

    result = GaussNewton(max_iterations=1).fit(exponential_decay, x, y, [1.0, 1.0, 0.0])

    assert not result.converged
    assert result.n_iterations == 1
    assert len(result.history) == 2


def test_rejects_mismatched_model_output():
    with pytest.raises(ValueError, match="predictions"):
        GaussNewton().fit(lambda p, x: np.zeros(3), np.arange(5.0), np.arange(5.0), [0.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"tolerance": -1.0},
        {"method": "sideways"},
        {"step": 0.0},
    ],
)
def test_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        GaussNewton(**kwargs)


def test_jacobian_rejects_invalid_method():
    with pytest.raises(ValueError, match="method"):
        finite_difference_jacobian(Polynomial(), [1.0], [0.0], method="sideways")
