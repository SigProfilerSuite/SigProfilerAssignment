"""Unit tests for the pure helper functions in SigProfilerAssignment.single_sample.
No genome reference required.
Run with: pytest tests/test_single_sample.py -v
"""
import numpy as np
import pytest

from SigProfilerAssignment.single_sample import (
    add_connected_sigs,
    constraints1,
    create_bounds,
    get_changed_background_sig_idx,
    parameterized_objective2_custom,
    roundConserveSum,
)


# ─── roundConserveSum ──────────────────────────────────────────────

class TestRoundConserveSum:
    def test_conserves_total(self):
        x = np.array([1.2, 2.3, 3.5])
        result = roundConserveSum(x)
        assert list(result) == [1.0, 2.0, 4.0]
        assert np.sum(result) == pytest.approx(round(np.sum(x)))

    def test_already_integers(self):
        x = np.array([1.0, 2.0, 3.0])
        result = roundConserveSum(x)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_single_element(self):
        # np.round uses round-half-to-even, so 2.5 rounds to 2.0, not 3.0.
        result = roundConserveSum(np.array([2.5]))
        assert list(result) == [2.0]

    def test_tie_residuals(self):
        result = roundConserveSum(np.array([0.5, 0.5]))
        assert np.sum(result) == 1.0


# ─── constraints1 ──────────────────────────────────────────────

class TestConstraints1:
    def test_matches_sample_total(self):
        # Result is the *total* sample sum minus the total of x, not per-element.
        samples = np.array([10, 20])
        assert constraints1(np.array([15, 15]), samples) == 0

    def test_under_sample_total(self):
        samples = np.array([10, 20])
        assert constraints1(np.array([5, 4]), samples) == 21

    def test_zero_samples(self):
        assert constraints1(np.array([0, 0]), np.array([0, 0])) == 0


# ─── create_bounds ──────────────────────────────────────────────

class TestCreateBounds:
    def test_basic_bounds(self):
        bounds = create_bounds([1], np.array([10, 20]), 3)
        assert bounds == [(0.0, 30.0), (0.0, 0.0), (0.0, 30.0)]

    def test_no_zeros(self):
        bounds = create_bounds([], np.array([5]), 2)
        assert bounds == [(0.0, 5.0), (0.0, 5.0)]

    def test_all_zeros(self):
        bounds = create_bounds([0, 1], np.array([5]), 2)
        assert bounds == [(0.0, 0.0), (0.0, 0.0)]


# ─── parameterized_objective2_custom ──────────────────────────────────────────────

class TestParameterizedObjective2Custom:
    def test_exact_reconstruction_is_zero(self):
        signatures = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([2.0, 3.0])
        samples = np.array([[2.0], [3.0]])
        assert parameterized_objective2_custom(x, signatures, samples) == pytest.approx(0.0)

    def test_falls_back_for_1d_samples(self):
        signatures = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([2.0, 3.0])
        samples = np.array([2.0, 3.0])
        # 1D samples hit the broadcasting branch (except clause), not an exact zero.
        assert parameterized_objective2_custom(x, signatures, samples) > 0.0

    def test_nonzero_residual(self):
        signatures = np.array([[1.0, 0.0], [0.0, 1.0]])
        x = np.array([2.0, 3.0])
        samples = np.array([[0.0], [0.0]])
        assert parameterized_objective2_custom(x, signatures, samples) == pytest.approx(
            np.linalg.norm([2.0, 3.0])
        )


# ─── get_changed_background_sig_idx ──────────────────────────────────────────────

class TestGetChangedBackgroundSigIdx:
    def test_reindexes_after_dropping_zero_exposures(self):
        exposures = [5, 0, 3, 7]
        background_sigs = [0, 3]  # values 5 and 7
        assert get_changed_background_sig_idx(exposures, background_sigs) == [0, 2]

    def test_background_sig_with_zero_exposure_is_dropped(self):
        exposures = [5, 0, 3, 0]
        background_sigs = [1, 3]  # both values are 0
        assert get_changed_background_sig_idx(exposures, background_sigs) == []

    def test_empty_background_sigs(self):
        assert get_changed_background_sig_idx([5, 3], []) == []


# ─── add_connected_sigs ──────────────────────────────────────────────

class TestAddConnectedSigs:
    def test_pulls_in_connected_signature(self):
        allsigids = ["SBS1", "SBS2", "SBS7a", "SBS13"]
        background_sigs = [1]  # SBS2, connected to SBS13
        assert add_connected_sigs(background_sigs, allsigids) == [1, 3]

    def test_no_connected_signature_present(self):
        allsigids = ["SBS1", "SBS3", "SBS5"]
        background_sigs = [0]  # SBS1 has no connected partners in this list
        assert add_connected_sigs(background_sigs, allsigids) == [0]

    def test_empty_background(self):
        allsigids = ["SBS1", "SBS2"]
        assert add_connected_sigs([], allsigids) == []
