"""Unit tests for the pure helper functions in SigProfilerAssignment.decompose_subroutines.
No genome reference required.
Run with: pytest tests/test_decompose_subroutines.py -v
"""
import numpy as np
import pytest

from SigProfilerAssignment.decompose_subroutines import (
    cor_sim,
    cos_sim,
    get_indeces,
    get_items_from_index,
    make_letter_ids,
    signature_plotting_text,
    union,
)


# ─── union ──────────────────────────────────────────────

class TestUnion:
    def test_disjoint_lists(self):
        assert sorted(union([1, 2], [3, 4])) == [1, 2, 3, 4]

    def test_overlapping_lists(self):
        assert sorted(union(["SBS1", "SBS2"], ["SBS2", "SBS3"])) == ["SBS1", "SBS2", "SBS3"]

    def test_empty_lists(self):
        assert union([], []) == []

    def test_one_empty_list(self):
        assert sorted(union(["A"], [])) == ["A"]


# ─── get_indeces ──────────────────────────────────────────────

class TestGetIndeces:
    def test_basic(self):
        x = ["SBS1", "SBS2", "SBS3", "SBS5", "SBS8", "SBS13", "SBS40"]
        y = ["SBS1", "SBS5"]
        assert get_indeces(x, y) == [0, 3]

    def test_preserves_query_order(self):
        x = ["SBS1", "SBS2", "SBS3"]
        y = ["SBS3", "SBS1"]
        assert get_indeces(x, y) == [2, 0]

    def test_missing_item_is_skipped(self):
        x = ["SBS1", "SBS2"]
        y = ["SBS1", "SBS99"]
        assert get_indeces(x, y) == [0]

    def test_empty_query(self):
        assert get_indeces(["SBS1", "SBS2"], []) == []


# ─── get_items_from_index ──────────────────────────────────────────────

class TestGetItemsFromIndex:
    def test_basic(self):
        x = ["SBS1", "SBS2", "SBS3", "SBS5"]
        y = [1, 3]
        assert get_items_from_index(x, y) == ["SBS2", "SBS5"]

    def test_out_of_range_index_is_skipped(self):
        x = ["SBS1", "SBS2"]
        y = [0, 5]
        assert get_items_from_index(x, y) == ["SBS1"]

    def test_empty_index_list(self):
        assert get_items_from_index(["SBS1"], []) == []


# ─── make_letter_ids ──────────────────────────────────────────────

class TestMakeLetterIds:
    def test_single_letter_range(self):
        ids = make_letter_ids(idlenth=3, mtype="SBS96")
        assert list(ids) == ["SBS96A", "SBS96B", "SBS96C"]

    def test_default_length(self):
        ids = make_letter_ids()
        assert len(ids) == 10

    def test_rolls_over_to_double_letters(self):
        ids = make_letter_ids(idlenth=27, mtype="SBS96")
        assert ids[26] == "SBS96AA"

    def test_zero_length(self):
        ids = make_letter_ids(idlenth=0, mtype="SBS96")
        assert len(ids) == 0


# ─── signature_plotting_text ──────────────────────────────────────────────

class TestSignaturePlottingText:
    def test_integer_type(self):
        result = signature_plotting_text([25, 75], "Sample1", "integer")
        assert result == ["Sample1: 25/25.0%", "Sample1: 75/75.0%"]

    def test_float_type(self):
        result = signature_plotting_text([1.234, 2.5], "Sample1", "float")
        assert result == ["Sample1: 1.23", "Sample1: 2.5"]

    def test_single_value_is_whole_total(self):
        result = signature_plotting_text([42], "Sample1", "integer")
        assert result == ["Sample1: 42/100.0%"]


# ─── cos_sim ──────────────────────────────────────────────

class TestCosSim:
    def test_identical_vectors(self):
        assert cos_sim([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cos_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cos_sim([0, 0, 0], [1, 2, 3]) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        assert cos_sim([0, 0], [0, 0]) == 0.0


# ─── cor_sim ──────────────────────────────────────────────

class TestCorSim:
    def test_perfectly_correlated_vectors(self):
        assert cor_sim([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        assert cor_sim([0, 0, 0], [1, 2, 3]) == 0.0

    def test_uncorrelated_constant_vector(self):
        # scipy's correlation distance is undefined (nan) for a constant,
        # non-zero vector; cor_sim only special-cases all-zero input.
        result = cor_sim([5, 5, 5], [1, 2, 3])
        assert np.isnan(result)
