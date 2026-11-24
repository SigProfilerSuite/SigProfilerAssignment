"""Unit tests for the functions in SigProfilerAssignment.single_sample.
No genome reference required (fitting tests use the COSMIC SBS96 reference
shipped with the package to build noise-free dummy samples).
Run with: pytest tests/test_single_sample.py -v
"""
import os

import numpy as np
import pandas as pd
import pytest

import SigProfilerAssignment as spa
from SigProfilerAssignment.single_sample import (
    add_connected_sigs,
    add_remove_signatures,
    constraints1,
    create_bounds,
    fit_signatures,
    get_changed_background_sig_idx,
    parameterized_objective2_custom,
    remove_all_single_signatures,
    roundConserveSum,
    same_active_signatures,
)

REF_SBS96 = os.path.join(
    os.path.dirname(spa.__file__),
    "data", "Reference_Signatures", "GRCh37", "COSMIC_v3.6_SBS_GRCh37.txt",
)


def _cosmic(names):
    """Columns of the packaged COSMIC v3.6 SBS96 reference (W matrix and names)."""
    ref = pd.read_csv(REF_SBS96, sep="\t", index_col=0)
    return ref[names].to_numpy(), list(names)


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


# ─── same_active_signatures ──────────────────────────────────────────────

class TestSameActiveSignatures:
    def test_different_sets_of_the_same_size_are_different(self):
        # the old check compared np.nonzero(a)[0].all() == np.nonzero(b)[0].all(), i.e. two booleans
        assert not same_active_signatures(np.array([0, 5, 7, 0]), np.array([0, 0, 7, 3]))

    def test_same_active_set_with_other_values_is_the_same(self):
        assert same_active_signatures(np.array([0, 5, 7]), np.array([0, 2, 9]))

    def test_different_sizes_are_different(self):
        assert not same_active_signatures(np.array([1, 5, 7]), np.array([0, 5, 7]))


# ─── fit_signatures ──────────────────────────────────────────────

class TestFitSignatures:
    def test_recovers_noise_free_mixture(self):
        W, _ = _cosmic(["SBS1", "SBS5", "SBS13"])
        exposures, _ = fit_signatures(W, W @ np.array([1000.0, 2000.0, 500.0]))
        assert np.allclose(exposures, [1000, 2000, 500], atol=1)

    def test_conserves_mutation_burden(self):
        W, _ = _cosmic(["SBS1", "SBS5", "SBS13"])
        genome = np.round(W @ np.array([300.0, 700.0, 50.0]))
        exposures, _ = fit_signatures(W, genome)
        assert np.sum(exposures) == round(np.sum(genome))

    def test_returns_integer_activities(self):
        W, _ = _cosmic(["SBS1", "SBS5"])
        exposures, _ = fit_signatures(W, np.round(W @ np.array([123.0, 456.0])))
        assert np.array_equal(exposures, np.round(exposures))

    def test_does_not_change_the_global_random_state(self):
        # the NNLS fit is deterministic and must not consume numbers from numpy's global RNG
        W, _ = _cosmic(["SBS1", "SBS5", "SBS13"])
        before = np.random.get_state()[1].copy()
        fit_signatures(W, np.round(W @ np.array([300.0, 700.0, 50.0])))
        assert np.array_equal(np.random.get_state()[1], before)

    def test_exact_reconstruction_has_zero_l2_error(self):
        W, _ = _cosmic(["SBS1", "SBS5"])
        _, error = fit_signatures(W, W @ np.array([100.0, 200.0]))
        assert error == pytest.approx(0.0, abs=1e-9)


# ─── remove_all_single_signatures ──────────────────────────────────────────────

class TestRemoveAllSingleSignatures:
    def test_single_active_signature_is_returned_unchanged(self):
        W, _ = _cosmic(["SBS1", "SBS5"])
        H = np.array([0.0, 500.0])
        exposures, _, _ = remove_all_single_signatures(W, H, W @ H, cutoff=0.01)
        assert list(exposures) == [0, 500]

    def test_removes_signature_with_negligible_contribution(self):
        W, _ = _cosmic(["SBS1", "SBS5", "SBS13"])
        genome = W @ np.array([1000.0, 2000.0, 0.0])
        H = np.array([1000.0, 1998.0, 2.0])
        exposures, _, _ = remove_all_single_signatures(W, H, genome, cutoff=0.01)
        assert exposures[2] == 0
        assert np.sum(exposures) == pytest.approx(np.sum(genome))

    def test_keeps_protected_signature_without_tied_activities(self):
        W, _ = _cosmic(["SBS13", "SBS5", "SBS40a"])
        H = np.array([12.0, 13.0, 899.0])
        exposures, _, _ = remove_all_single_signatures(W, H, W @ H, cutoff=0.01, background_sigs=[1])
        assert exposures[1] != 0


# ─── add_remove_signatures ──────────────────────────────────────────────

class TestAddRemoveSignatures:
    NAMES = ["SBS1", "SBS2", "SBS5", "SBS13", "SBS18", "SBS40a"]

    def _run(self, tmp_path, mixture):
        W, names = _cosmic(self.NAMES)
        sample = np.round(W @ np.array([mixture.get(n, 0.0) for n in names]))
        result = add_remove_signatures(
            W, sample, allsigids=names, connected_sigs=False,
            directory=str(tmp_path / "assignment_log.txt"),
        )
        return result, sample, names

    def test_recovers_signature_set_of_noise_free_sample(self, tmp_path):
        mixture = {"SBS1": 1000.0, "SBS5": 2000.0, "SBS13": 500.0}
        (_, activities, *_), _, names = self._run(tmp_path, mixture)
        selected = {n for n, a in zip(names, activities) if a != 0}
        assert selected == set(mixture)

    def test_conserves_mutation_burden(self, tmp_path):
        (_, activities, *_), sample, _ = self._run(tmp_path, {"SBS1": 400.0, "SBS18": 900.0})
        assert np.sum(activities) == round(np.sum(sample))

    def test_writes_assignment_log(self, tmp_path):
        self._run(tmp_path, {"SBS5": 800.0})
        log = (tmp_path / "assignment_log.txt").read_text()
        assert "Final Composition" in log
