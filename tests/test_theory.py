import numpy as np

from lpui.theory import (
    minimax_rate,
    oracle_exposure_ht_mse,
    privacy_signal_squared,
)


def test_rate_separates_interference_and_privacy_bottlenecks():
    loose_privacy = minimax_rate(cluster_count=100, cluster_size=20, epsilon=20.0)
    tight_privacy = minimax_rate(cluster_count=100, cluster_size=20, epsilon=0.1)

    assert np.isclose(loose_privacy.interference_term, 0.01)
    assert loose_privacy.privacy_term < loose_privacy.interference_term
    assert tight_privacy.privacy_term > tight_privacy.interference_term
    assert tight_privacy.total < 1.0
    extreme_privacy = minimax_rate(cluster_count=100, cluster_size=20, epsilon=0.001)
    assert extreme_privacy.total == 1.0


def test_privacy_signal_has_correct_small_epsilon_scaling():
    epsilon = np.array([1e-3, 2e-3, 4e-3])
    ratios = privacy_signal_squared(epsilon) / epsilon**2
    assert np.allclose(ratios, 0.25, rtol=1e-5)


def test_oracle_exposure_ht_mse_matches_direct_moment_enumeration():
    sample_size = 250
    exposure_probability = 0.125
    epsilon = 0.9
    target_mean = 0.6
    baseline = np.exp(-epsilon) / (1.0 + np.exp(-epsilon))
    signal = np.tanh(epsilon / 2.0)
    released_one_probability = baseline + signal * target_mean
    support = np.array([0.0, -baseline, 1.0 - baseline])
    probabilities = np.array(
        [
            1.0 - exposure_probability,
            exposure_probability * (1.0 - released_one_probability),
            exposure_probability * released_one_probability,
        ]
    )
    per_user_variance = np.dot(probabilities, support**2) - (
        np.dot(probabilities, support)
    ) ** 2
    direct = per_user_variance / (
        sample_size * exposure_probability**2 * signal**2
    )

    assert np.isclose(
        oracle_exposure_ht_mse(
            sample_size, exposure_probability, epsilon, target_mean
        ),
        direct,
    )
