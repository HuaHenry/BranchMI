import math

import pytest

from branchmi_pilot.scoring import generalized_js, normalized_js


def test_js_is_zero_for_identical_distributions():
    distributions = [{0: 0.25, 1: 0.75}, {0: 0.25, 1: 0.75}]
    assert generalized_js(distributions, [0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)


def test_js_reaches_log_two_for_disjoint_uniform_branches():
    value = generalized_js([{0: 1.0}, {1: 1.0}], [0.5, 0.5])
    assert value == pytest.approx(math.log(2), rel=1e-10)
    assert normalized_js(value, [0.5, 0.5]) == pytest.approx(1.0)


def test_weighted_js_matches_weight_entropy_for_point_masses():
    weights = [0.8, 0.2]
    value = generalized_js([{0: 1.0}, {1: 1.0}], weights)
    expected = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    assert value == pytest.approx(expected)

