import numpy as np
import pytest

from gauss_newton import Polynomial, rmse


def test_polynomial_evaluates_coefficients_in_increasing_order():
    x = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(Polynomial()([1.0, -2.0, 0.5], x), 1.0 - 2.0 * x + 0.5 * x**2)


def test_polynomial_matches_numpy():
    rng = np.random.default_rng(0)
    params, x = rng.normal(size=6), rng.uniform(-2, 2, size=50)
    np.testing.assert_allclose(Polynomial()(params, x), np.polynomial.polynomial.polyval(x, params))


def test_polynomial_accepts_column_vector_params_and_preserves_x_shape():
    x = np.linspace(-1, 1, 5).reshape(1, -1)
    result = Polynomial()(np.array([[1.0], [2.0]]), x)
    assert result.shape == x.shape
    np.testing.assert_allclose(result, 1.0 + 2.0 * x)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ([1.0, -2.0, 0.5], "1.00 - 2.00x + 0.50x^2"),
        ([-1.5], "-1.50"),
        ([0.0, 3.0], "0.00 + 3.00x"),
    ],
)
def test_polynomial_to_string(params, expected):
    assert Polynomial.to_string(params) == expected


def test_rmse():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert rmse([0.0, 0.0], [3.0, -3.0]) == pytest.approx(3.0)
