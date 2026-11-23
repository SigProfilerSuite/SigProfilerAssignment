"""Unit tests for the pure helper functions in SigProfilerAssignment.decompose_subroutines.
No genome reference required.
Run with: pytest tests/test_decompose_subroutines.py -v
"""
import numpy as np
import pandas as pd
import pytest

from SigProfilerAssignment.decompose_subroutines import (
    collapse_not_applicable_message,
    collapse_to_sbs96,
    cor_sim,
    cos_sim,
    get_indeces,
    getProcessAvg,
    is_extended_sbs,
    get_items_from_index,
    make_letter_ids,
    probabilities,
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


# ─── is_extended_sbs / collapse_to_sbs96 ──────────────────────────────────────────────

def _context(plot_type):
    from sigProfilerPlotting import sigProfilerPlotting as sigPlot
    return sigPlot.get_context_reference(plot_type)


class TestIsExtendedSbs:
    @pytest.mark.parametrize("plot_type", ["288", "384", "1536", "4608", "6144"])
    def test_extended_sbs_contexts(self, plot_type):
        assert is_extended_sbs(_context(plot_type))

    def test_sbs96_is_not_extended(self):
        assert not is_extended_sbs(_context("96"))

    @pytest.mark.parametrize("plot_type", ["78", "83"])
    def test_dbs_and_id_are_not_extended_sbs(self, plot_type):
        assert not is_extended_sbs(_context(plot_type))

    def test_arbitrary_labels_are_not_extended_sbs(self):
        assert not is_extended_sbs(["CTX%03d" % i for i in range(200)])


class TestCollapseToSbs96:
    @pytest.mark.parametrize("plot_type", ["288", "384", "1536", "4608", "6144"])
    def test_gives_the_96_contexts_in_canonical_order(self, plot_type):
        labels = _context(plot_type)
        matrix = pd.DataFrame({"S1": np.ones(len(labels))}, index=labels)
        collapsed = collapse_to_sbs96(matrix)
        assert list(collapsed.index) == _context("96")
        assert collapsed["S1"].sum() == len(labels)

    def test_sums_rows_by_central_trinucleotide(self):
        labels = _context("288")
        matrix = pd.DataFrame({"S1": np.arange(len(labels), dtype=float)}, index=labels)
        collapsed = collapse_to_sbs96(matrix)
        expected = sum(matrix.loc[lab, "S1"] for lab in labels if lab.endswith("A[C>A]A"))
        assert collapsed.loc["A[C>A]A", "S1"] == expected

    def test_label_order_does_not_matter(self):
        labels = _context("1536")
        matrix = pd.DataFrame({"S1": np.arange(len(labels), dtype=float)}, index=labels)
        shuffled = matrix.iloc[::-1]
        pd.testing.assert_frame_equal(collapse_to_sbs96(shuffled), collapse_to_sbs96(matrix))

    def test_non_sbs_label_raises_value_error(self):
        matrix = pd.DataFrame({"S1": [1.0, 2.0]}, index=["A[C>A]A", "DBS_AC>CA"])
        with pytest.raises(ValueError, match="DBS_AC>CA"):
            collapse_to_sbs96(matrix)

    def test_purine_centred_contexts_raise_value_error(self):
        # same mutations written from the purine strand: A[C>A]G becomes C[G>T]T
        comp = str.maketrans("ACGT", "TGCA")
        labels = []
        for lab in _context("288"):
            strand, ctx = lab.split(":")
            labels.append("%s:%s[%s>%s]%s" % (strand, ctx[6].translate(comp), ctx[2].translate(comp),
                                               ctx[4].translate(comp), ctx[0].translate(comp)))
        matrix = pd.DataFrame({"S1": np.ones(len(labels))}, index=labels)
        with pytest.raises(ValueError, match=r"not standard SBS96 contexts \(e\.g\. '.\[[AG]>.\].'\)\. "
                                             r"Write substitutions from the pyrimidine base"):
            collapse_to_sbs96(matrix)

    def test_incomplete_contexts_raise_value_error(self):
        # drop every strand variant of one trinucleotide, so only 95 contexts remain
        labels = [lab for lab in _context("288") if not lab.endswith("T[T>G]T")]
        matrix = pd.DataFrame({"S1": np.ones(len(labels))}, index=labels)
        with pytest.raises(ValueError, match="95"):
            collapse_to_sbs96(matrix)


class TestCollapseNotApplicableMessage:
    PREFIX = ("collapse_to_SBS96=True only applies to SBS inputs with extended contexts "
              "(e.g. SBS288, SBS1536, SBS4608); ")

    @pytest.mark.parametrize("rows,name", [(78, "DBS78"), (83, "ID83"), (48, "CNV48"), (32, "SV32")])
    def test_known_context_is_named(self, rows, name):
        assert collapse_not_applicable_message(rows) == (
            self.PREFIX + "the input is %s, so it is analysed without collapsing." % name)

    def test_unknown_context_gives_row_count(self):
        assert collapse_not_applicable_message(10) == (
            self.PREFIX + "the input has 10 rows (mutation contexts), so it is analysed without collapsing.")


# ─── getProcessAvg ──────────────────────────────────────────────

class TestGetProcessAvg:
    def test_unsupported_context_raises_value_error(self):
        # audit SPA-2026-11 #19: used to return the input itself as the signature database
        samples = pd.DataFrame(np.ones((95, 2)), index=["C%02d" % i for i in range(95)], columns=["S1", "S2"])
        with pytest.raises(ValueError, match="Input samples have 95 rows"):
            getProcessAvg(samples)


# ─── probabilities ──────────────────────────────────────────────

class TestProbabilities:
    W = np.array([[0.5, 0.1], [0.3, 0.2], [0.2, 0.7]])
    H = np.array([[10.0, 0.0], [30.0, 5.0]])

    def _probs(self):
        return probabilities(self.W, self.H, ["c1", "c2", "c3"], ["sigA", "sigB"], ["S1", "S2"])

    def test_one_row_per_context_and_sample(self):
        probs = self._probs()
        assert len(probs) == 6
        assert list(probs.columns) == ["Sample Names", "MutationType", "sigA", "sigB"]
        assert list(probs["Sample Names"]) == ["S1"] * 3 + ["S2"] * 3

    def test_rows_with_reconstructed_mass_sum_to_one(self):
        probs = self._probs()
        assert np.allclose(probs[["sigA", "sigB"]].sum(axis=1), 1.0)

    def test_matches_definition(self):
        probs = self._probs()
        # S1, context c2: sigA 0.3*10 = 3, sigB 0.2*30 = 6 -> 1/3 and 2/3
        row = probs[(probs["Sample Names"] == "S1") & (probs["MutationType"] == "c2")]
        assert row["sigA"].item() == pytest.approx(1 / 3)
        assert row["sigB"].item() == pytest.approx(2 / 3)

    def test_signature_without_activity_gets_zero(self):
        probs = self._probs()
        assert (probs.loc[probs["Sample Names"] == "S2", "sigA"] == 0).all()
