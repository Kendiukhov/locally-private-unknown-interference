import numpy as np

from lpui.mechanisms import debias_one_bit, one_bit_probability, privatize_one_bit


def test_one_bit_channel_is_exactly_epsilon_ldp_at_extreme_inputs():
    epsilon = 0.7
    probability_at_zero = one_bit_probability(np.array([0.0]), epsilon)[0]
    probability_at_one = one_bit_probability(np.array([1.0]), epsilon)[0]

    assert np.isclose(probability_at_one / probability_at_zero, np.exp(epsilon))
    assert np.isclose(
        (1.0 - probability_at_zero) / (1.0 - probability_at_one),
        np.exp(epsilon),
    )


def test_debiased_one_bit_release_is_unbiased():
    rng = np.random.default_rng(912)
    outcomes = np.repeat(np.array([0.1, 0.5, 0.9]), 300_000)
    releases = privatize_one_bit(outcomes, epsilon=1.1, rng=rng)
    decoded = debias_one_bit(releases, epsilon=1.1).reshape(3, -1)

    assert np.allclose(decoded.mean(axis=1), [0.1, 0.5, 0.9], atol=0.008)


def test_one_bit_mechanism_rejects_out_of_range_outcomes():
    rng = np.random.default_rng(7)
    with np.testing.assert_raises_regex(ValueError, r"\[0, 1\]"):
        privatize_one_bit(np.array([-0.01, 0.5]), epsilon=1.0, rng=rng)


def test_one_bit_probability_is_numerically_stable_at_large_epsilon():
    with np.errstate(over="raise"):
        probabilities = one_bit_probability(np.array([0.0, 0.5, 1.0]), 1_000.0)

    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities, [0.0, 0.5, 1.0])
