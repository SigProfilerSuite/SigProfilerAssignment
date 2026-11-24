"""Characterization tests for spa_analyze through the public Analyzer API
(cosmic_fit, denovo_fit, decompose_fit) with SBS96 matrix input.

Inputs are dummy matrices built inside the tests from the COSMIC v3.6 SBS96
reference shipped with the package (noise-free mixtures of SBS1, SBS5 and SBS13),
so no genome reference, network or real data is needed. Outputs go to tmp_path.
The tests lock the current behaviour on valid input: selected signatures,
activities (within +-2 mutations to tolerate platform rounding), mutation-burden
conservation, output files and finite mutation-type probabilities.

Run with: pytest tests/test_decomposition_files.py -v
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest

import SigProfilerAssignment as spa
from SigProfilerAssignment import Analyzer as Analyze

REF_SBS96 = os.path.join(
    os.path.dirname(spa.__file__),
    "data", "Reference_Signatures", "GRCh37", "COSMIC_v3.6_SBS_GRCh37.txt",
)
# Dummy samples: noise-free mixtures (mutations contributed by each signature).
MIXTURES = {
    "S1": {"SBS1": 1000, "SBS5": 2000, "SBS13": 500},
    "S2": {"SBS1": 300, "SBS5": 1500},
    "S3": {"SBS1": 800, "SBS5": 600, "SBS13": 1200},
}
TOLERANCE = 2  # mutations


def _cosmic():
    return pd.read_csv(REF_SBS96, sep="\t", index_col=0)


def _dummy_samples():
    cosmic = _cosmic()
    cols = {}
    for name, mix in MIXTURES.items():
        profile = sum(n * cosmic[sig] for sig, n in mix.items())
        cols[name] = np.round(profile).astype(int)
    samples = pd.DataFrame(cols, index=cosmic.index)
    samples.index.name = "MutationType"
    return samples


def _dummy_denovo_signatures():
    cosmic = _cosmic()
    sigs = pd.DataFrame(
        {"SBS96A": 0.6 * cosmic["SBS1"] + 0.4 * cosmic["SBS5"], "SBS96B": cosmic["SBS13"]},
        index=cosmic.index,
    )
    sigs.index.name = "MutationType"
    return sigs


def _run(fn, output, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        fn(output=str(output), make_plots=False, export_probabilities=True, cpu=1, **kwargs)
    return output


def _activities(output, solution, prefix):
    path = os.path.join(output, solution, "Activities", prefix + "_Activities.txt")
    return pd.read_csv(path, sep="\t", index_col=0)


def _assert_close(activities, expected):
    for sample, sigs in expected.items():
        active = activities.loc[sample]
        active = active[active != 0]
        assert set(active.index) == set(sigs), sample
        for sig, n in sigs.items():
            assert abs(active[sig] - n) <= TOLERANCE, (sample, sig, active[sig], n)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("spa_inputs")
    samples = _dummy_samples()
    samples_path = d / "samples.txt"
    samples.to_csv(samples_path, sep="\t")
    denovo_path = d / "denovo_signatures.txt"
    _dummy_denovo_signatures().to_csv(denovo_path, sep="\t")
    db = _cosmic()[["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]]
    db_path = d / "custom_db_canonical.txt"
    db.to_csv(db_path, sep="\t")
    return {"samples": samples, "samples_path": str(samples_path),
            "denovo_path": str(denovo_path), "db_path": str(db_path)}


@pytest.fixture(scope="module")
def cosmic_run(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("cosmic_fit")
    return _run(Analyze.cosmic_fit, out, samples=inputs["samples_path"])


@pytest.fixture(scope="module")
def denovo_run(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("denovo_fit")
    return _run(Analyze.denovo_fit, out, samples=inputs["samples_path"],
                signatures=inputs["denovo_path"])


@pytest.fixture(scope="module")
def decompose_run(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("decompose_fit")
    return _run(Analyze.decompose_fit, out, samples=inputs["samples_path"],
                signatures=inputs["denovo_path"])


# ─── cosmic_fit ──────────────────────────────────────────────

class TestCosmicFit:
    def test_recovers_dummy_mixtures(self, cosmic_run):
        acts = _activities(cosmic_run, "Assignment_Solution", "Assignment_Solution")
        _assert_close(acts, MIXTURES)

    def test_activities_conserve_mutation_burden(self, cosmic_run, inputs):
        acts = _activities(cosmic_run, "Assignment_Solution", "Assignment_Solution")
        burden = inputs["samples"].sum()
        assert (acts.sum(axis=1) == burden.loc[acts.index]).all()

    def test_writes_expected_files(self, cosmic_run):
        for rel in (
            "JOB_METADATA_SPA.txt",
            "Assignment_Solution/Activities/Assignment_Solution_Activities.txt",
            "Assignment_Solution/Activities/Decomposed_MutationType_Probabilities.txt",
            "Assignment_Solution/Signatures/Assignment_Solution_Signatures.txt",
            "Assignment_Solution/Solution_Stats/Assignment_Solution_Samples_Stats.txt",
            "Assignment_Solution/Solution_Stats/Assignment_Solution_Signature_Assignment_log.txt",
        ):
            assert os.path.isfile(os.path.join(cosmic_run, rel)), rel

    def test_probabilities_are_finite_and_sum_to_one(self, cosmic_run):
        path = os.path.join(cosmic_run, "Assignment_Solution", "Activities",
                            "Decomposed_MutationType_Probabilities.txt")
        probs = pd.read_csv(path, sep="\t")
        values = probs.drop(columns=["Sample Names", "MutationType"])
        assert np.isfinite(values.to_numpy()).all()
        assert np.allclose(values.sum(axis=1), 1.0)

    def test_does_not_change_the_global_random_state(self, inputs, tmp_path):
        # the assignment is deterministic and must not consume numbers from numpy's global RNG
        before = np.random.get_state()[1].copy()
        _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"])
        assert np.array_equal(np.random.get_state()[1], before)

    def test_dataframe_input_matches_file_input(self, cosmic_run, inputs, tmp_path):
        out = _run(Analyze.cosmic_fit, tmp_path / "df", samples=inputs["samples"])
        from_df = _activities(out, "Assignment_Solution", "Assignment_Solution")
        from_file = _activities(cosmic_run, "Assignment_Solution", "Assignment_Solution")
        pd.testing.assert_frame_equal(from_df, from_file)

    def test_custom_signature_database_in_canonical_order(self, inputs, tmp_path):
        out = _run(Analyze.cosmic_fit, tmp_path / "db", samples=inputs["samples_path"],
                   signature_database=inputs["db_path"])
        acts = _activities(out, "Assignment_Solution", "Assignment_Solution")
        _assert_close(acts, MIXTURES)


# ─── custom contexts with a user signature database ─────────────────────────────

def _custom_context_inputs(tmp_path, contexts, n_sigs=3, scale=1):
    """Dummy signatures (local seeded generator) over `contexts` and noise-free samples.
    `scale` multiplies the burdens so that wide contexts (e.g. 4608 rows) still have
    several mutations per context and integer rounding does not distort the profile."""
    rng = np.random.default_rng(0)
    names = ["SIG_%s" % chr(65 + k) for k in range(n_sigs)]
    sigs = pd.DataFrame(rng.dirichlet(np.full(len(contexts), 0.5), size=n_sigs).T,
                        index=contexts, columns=names)
    sigs.index.name = "MutationType"
    mixtures = {s: {k: n * scale for k, n in mix.items()} for s, mix in {
        "S1": {"SIG_A": 1500, "SIG_B": 700},
        "S2": {"SIG_B": 400, "SIG_C": 2200},
        "S3": {"SIG_A": 900, "SIG_B": 900, "SIG_C": 900}}.items()}
    samples = pd.DataFrame(
        {s: np.round(sum(n * sigs[k] for k, n in mix.items())).astype(int) for s, mix in mixtures.items()},
        index=contexts)
    samples.index.name = "MutationType"
    db_path, samples_path = tmp_path / "db.txt", tmp_path / "samples.txt"
    sigs.to_csv(db_path, sep="\t")
    samples.to_csv(samples_path, sep="\t")
    return str(samples_path), str(db_path), mixtures


class TestCustomContext:
    def test_sbs4608_with_custom_database_without_collapse(self, tmp_path):
        from sigProfilerPlotting import sigProfilerPlotting as sigPlot
        contexts = sigPlot.get_context_reference("4608")
        samples, db, mixtures = _custom_context_inputs(tmp_path, contexts, scale=100)
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=samples,
                   signature_database=db, collapse_to_SBS96=False)
        acts = _activities(out, "Assignment_Solution", "Assignment_Solution")
        for sample, sigs in mixtures.items():
            active = acts.loc[sample][acts.loc[sample] != 0]
            assert set(active.index) == set(sigs), sample
            for sig, n in sigs.items():
                assert abs(active[sig] - n) <= 0.005 * n, (sample, sig, active[sig], n)

    def test_arbitrary_context_with_custom_database_without_collapse(self, tmp_path):
        contexts = ["CTX%02d" % i for i in range(10)]
        samples, db, mixtures = _custom_context_inputs(tmp_path, contexts)
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=samples,
                   signature_database=db, collapse_to_SBS96=False)
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), mixtures)


# ─── collapse_to_SBS96 ──────────────────────────────────────────────

def _expand_sbs96(df96, context, counts):
    """Spread every SBS96 row over its extended-context rows (e.g. SBS288, SBS1536).
    counts=True splits integer counts exactly (collapsing gives back df96);
    counts=False splits probabilities evenly."""
    from sigProfilerPlotting import sigProfilerPlotting as sigPlot
    labels = sigPlot.get_context_reference(context)
    groups = {}
    for lab in labels:
        i, j = lab.index("["), lab.index("]")
        groups.setdefault(lab[i - 1:j + 2], []).append(lab)
    rows = {}
    for key, members in groups.items():
        k = len(members)
        for col in df96.columns:
            v = df96.loc[key, col]
            if counts:
                base, rem = divmod(int(v), k)
                parts = [base + (1 if n < rem else 0) for n in range(k)]
            else:
                parts = [v / k] * k
            for lab, part in zip(members, parts):
                rows.setdefault(lab, {})[col] = part
    out = pd.DataFrame.from_dict(rows, orient="index").loc[labels, list(df96.columns)]
    out.index.name = "MutationType"
    return out


class TestCollapseToSBS96:
    @pytest.mark.parametrize("context", ["288", "1536"])
    def test_extended_samples_collapse_to_the_sbs96_fit(self, context, cosmic_run, inputs, tmp_path):
        path = tmp_path / ("samples%s.txt" % context)
        _expand_sbs96(inputs["samples"], context, counts=True).to_csv(path, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path))
        expected = _activities(cosmic_run, "Assignment_Solution", "Assignment_Solution")
        pd.testing.assert_frame_equal(_activities(out, "Assignment_Solution", "Assignment_Solution"), expected)

    def test_decompose_sbs288_collapses_to_the_sbs96_fit(self, decompose_run, inputs, tmp_path):
        samples = tmp_path / "samples288.txt"
        denovo = tmp_path / "denovo288.txt"
        _expand_sbs96(inputs["samples"], "288", counts=True).to_csv(samples, sep="\t")
        _expand_sbs96(_dummy_denovo_signatures(), "288", counts=False).to_csv(denovo, sep="\t")
        out = _run(Analyze.decompose_fit, tmp_path / "out", samples=str(samples), signatures=str(denovo))
        expected = _activities(decompose_run, "Decompose_Solution", "Decompose_Solution")
        pd.testing.assert_frame_equal(_activities(out, "Decompose_Solution", "Decompose_Solution"), expected)

    # samples and database are collapsed separately, each from its own labels
    @pytest.mark.parametrize("context", ["288", "1536", "4608"])
    def test_extended_samples_and_custom_database_collapse_separately(self, context, inputs, tmp_path):
        samples, db = tmp_path / "samples.txt", tmp_path / "db.txt"
        _expand_sbs96(inputs["samples"], context, counts=True).to_csv(samples, sep="\t")
        _expand_sbs96(_cosmic()[["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]], context, counts=False).to_csv(db, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(samples), signature_database=str(db))
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), MIXTURES)

    # any SBS context of the form W[X>Y]Z collapses, not only SBS288 and SBS1536
    @pytest.mark.parametrize("context", ["384", "4608", "6144"])
    def test_other_extended_samples_collapse_to_the_sbs96_fit(self, context, cosmic_run, inputs, tmp_path):
        path = tmp_path / ("samples%s.txt" % context)
        _expand_sbs96(inputs["samples"], context, counts=True).to_csv(path, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path))
        expected = _activities(cosmic_run, "Assignment_Solution", "Assignment_Solution")
        pd.testing.assert_frame_equal(_activities(out, "Assignment_Solution", "Assignment_Solution"), expected)

    def test_sbs288_without_collapse_uses_the_packaged_v32_reference(self, inputs, tmp_path):
        path = tmp_path / "samples288.txt"
        _expand_sbs96(inputs["samples"], "288", counts=True).to_csv(path, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path), collapse_to_SBS96=False,
                   cosmic_version=3.2)
        acts = _activities(out, "Assignment_Solution", "Assignment_Solution")
        assert all(c.startswith("SBS288_Ref_") for c in acts.columns)
        assert (acts.sum(axis=1) == inputs["samples"].sum()).all()

    # SBS288 / SBS1536 have no packaged COSMIC reference (except SBS288 v3.2)
    @pytest.mark.parametrize("context,version", [("288", 3.6), ("1536", 3.6), ("1536", 3.2)])
    def test_extended_samples_without_reference_raise(self, context, version, inputs, tmp_path):
        path = tmp_path / ("samples%s.txt" % context)
        _expand_sbs96(inputs["samples"], context, counts=True).to_csv(path, sep="\t")
        expected = (r"^No COSMIC reference signatures are available for SBS{} \(cosmic_version {}\)\. SBS288 and "
                    r"SBS1536 inputs need a user-supplied signature_database; alternatively set "
                    r"collapse_to_SBS96=True to use the SBS96 COSMIC signatures\.$").format(context, version)
        with pytest.raises(ValueError, match=expected):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path), collapse_to_SBS96=False,
                 cosmic_version=version)

    def test_decompose_with_sbs1536_custom_database_collapsed(self, inputs, tmp_path):
        db = tmp_path / "db1536.txt"
        _expand_sbs96(_cosmic()[["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]], "1536", counts=False).to_csv(db, sep="\t")
        out = _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"],
                   signatures=inputs["denovo_path"], signature_database=str(db))
        table = pd.read_csv(os.path.join(out, "Decompose_Solution", "De_Novo_map_to_COSMIC_SBS96.csv"),
                            skipinitialspace=True)
        mapping = dict(zip(table.iloc[:, 0], table.iloc[:, 1]))
        assert mapping["Signature 96-SBS96A"] == "Signature SBS1 (60.00%) & Signature SBS5 (40.00%)"
        assert mapping["Signature 96-SBS96B"] == "Signature SBS13 (100.00%)"


# ─── collapse_to_SBS96 on contexts that cannot be collapsed ─────────────────────

def _run_capture(fn, output, **kwargs):
    """Like _run, but also returns what the run printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(output=str(output), make_plots=False, export_probabilities=True, cpu=1, **kwargs)
    return output, buf.getvalue()


def _dummy_dbs_inputs(tmp_path):
    ref = pd.read_csv(REF_SBS96.replace("_SBS_", "_DBS_"), sep="\t", index_col=0)
    mixtures = {"S1": {"DBS2": 300, "DBS4": 200}, "S2": {"DBS2": 150, "DBS11": 250}}
    samples = pd.DataFrame(
        {s: np.round(sum(n * ref[k] for k, n in mix.items())).astype(int) for s, mix in mixtures.items()},
        index=ref.index)
    samples.index.name = "MutationType"
    denovo = pd.DataFrame({"DBS78A": 0.5 * ref["DBS2"] + 0.5 * ref["DBS4"]}, index=ref.index)
    denovo.index.name = "MutationType"
    samples_path, denovo_path = tmp_path / "dbs_samples.txt", tmp_path / "dbs_denovo.txt"
    samples.to_csv(samples_path, sep="\t")
    denovo.to_csv(denovo_path, sep="\t")
    return str(samples_path), str(denovo_path), mixtures


class TestCollapseNotApplicable:
    MESSAGE = ("collapse_to_SBS96=True only applies to SBS inputs with extended contexts "
               "(e.g. SBS288, SBS1536, SBS4608); ")

    def test_dbs78_with_default_collapse_runs_and_warns(self, tmp_path):
        samples, _, mixtures = _dummy_dbs_inputs(tmp_path)
        out, printed = _run_capture(Analyze.cosmic_fit, tmp_path / "out", samples=samples)
        warning = self.MESSAGE + "the input is DBS78, so it is analysed without collapsing."
        assert warning in printed
        assert warning in open(os.path.join(out, "JOB_METADATA_SPA.txt")).read()
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), mixtures)

    def test_decompose_dbs78_with_default_collapse_warns(self, tmp_path):
        samples, denovo, _ = _dummy_dbs_inputs(tmp_path)
        _, printed = _run_capture(Analyze.decompose_fit, tmp_path / "out", samples=samples, signatures=denovo)
        assert self.MESSAGE + "the input is DBS78, so it is analysed without collapsing." in printed

    def test_same_custom_context_with_collapse_runs_and_warns(self, tmp_path):
        # used to stop with "Signatures Database and Samples are of same context type ..."
        samples, db, mixtures = _custom_context_inputs(tmp_path, ["CTX%02d" % i for i in range(10)])
        out, printed = _run_capture(Analyze.cosmic_fit, tmp_path / "out", samples=samples, signature_database=db)
        assert (self.MESSAGE + "the input has 10 rows (mutation contexts), so it is analysed "
                "without collapsing.") in printed
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), mixtures)

    def test_sbs96_input_does_not_warn(self, inputs, tmp_path):
        _, printed = _run_capture(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"])
        assert "collapse_to_SBS96=True only applies" not in printed


# ─── signature rows are matched to the samples by label ─────────────────────────

def _shuffled(df, seed):
    return df.sample(frac=1, random_state=seed)


class TestSignatureRowAlignment:
    # signature matrices used to be used by row position
    DB_SIGS = ["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]

    def test_cosmic_fit_shuffled_signature_database(self, inputs, tmp_path):
        db = tmp_path / "db_shuffled.txt"
        _shuffled(_cosmic()[self.DB_SIGS], 7).to_csv(db, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], signature_database=str(db))
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), MIXTURES)

    def test_denovo_fit_shuffled_signatures(self, denovo_run, inputs, tmp_path):
        sigs = tmp_path / "denovo_shuffled.txt"
        _shuffled(_dummy_denovo_signatures(), 3).to_csv(sigs, sep="\t")
        out = _run(Analyze.denovo_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=str(sigs))
        pd.testing.assert_frame_equal(_activities(out, "De_Novo_Solution", "De_Novo"),
                                      _activities(denovo_run, "De_Novo_Solution", "De_Novo"))

    def test_decompose_fit_shuffled_signatures(self, decompose_run, inputs, tmp_path):
        sigs = tmp_path / "denovo_shuffled.txt"
        _shuffled(_dummy_denovo_signatures(), 3).to_csv(sigs, sep="\t")
        out = _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=str(sigs))
        pd.testing.assert_frame_equal(_activities(out, "Decompose_Solution", "Decompose_Solution"),
                                      _activities(decompose_run, "Decompose_Solution", "Decompose_Solution"))

    def test_decompose_fit_shuffled_signature_database(self, inputs, tmp_path):
        db = tmp_path / "db_shuffled.txt"
        _shuffled(_cosmic()[self.DB_SIGS], 5).to_csv(db, sep="\t")
        out = _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"],
                   signatures=inputs["denovo_path"], signature_database=str(db))
        table = pd.read_csv(os.path.join(out, "Decompose_Solution", "De_Novo_map_to_COSMIC_SBS96.csv"),
                            skipinitialspace=True)
        assert dict(zip(table.iloc[:, 0], table.iloc[:, 1]))["Signature 96-SBS96A"] == \
            "Signature SBS1 (60.00%) & Signature SBS5 (40.00%)"

    def test_signature_database_with_duplicated_context_raises(self, inputs, tmp_path):
        db = _cosmic()[self.DB_SIGS].copy()
        db.index = [db.index[0]] + list(db.index[1:-1]) + [db.index[0]]
        path = tmp_path / "db_dup.txt"
        db.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=r"^signature_database has duplicated mutation contexts: A\[C>A\]A\. "
                                             r"Each mutation context must appear only once\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], signature_database=str(path))

    def test_signature_database_with_other_contexts_raises(self, inputs, tmp_path):
        db = _cosmic()[self.DB_SIGS].copy()
        db.index = list(db.index[:-1]) + ["X[C>A]A"]
        path = tmp_path / "db_other.txt"
        db.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=r"^signature_database and the input samples have different mutation "
                                             r"contexts \(missing in signature_database: T\[T>G\]T; not in the "
                                             r"samples: X\[C>A\]A\)\. Samples and signatures must have the same "
                                             r"mutation contexts\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], signature_database=str(path))

    def test_signatures_with_duplicated_context_raises(self, inputs, tmp_path):
        sigs = _dummy_denovo_signatures()
        sigs.index = [sigs.index[0]] + list(sigs.index[1:-1]) + [sigs.index[0]]
        path = tmp_path / "denovo_dup.txt"
        sigs.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=r"^The matrix passed in 'signatures' has duplicated mutation contexts: "
                                             r"A\[C>A\]A\. Each mutation context must appear only once\.$"):
            _run(Analyze.denovo_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=str(path))

    def test_signatures_with_other_contexts_raises(self, inputs, tmp_path):
        sigs = _dummy_denovo_signatures()
        sigs.index = list(sigs.index[:-1]) + ["X[C>A]A"]
        path = tmp_path / "denovo_other.txt"
        sigs.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=r"^The matrix passed in 'signatures' and the input samples have "
                                             r"different mutation contexts \(missing in 'signatures': T\[T>G\]T; "
                                             r"not in the samples: X\[C>A\]A\)\. Samples and signatures must have "
                                             r"the same mutation contexts\.$"):
            _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=str(path))


# ─── input validation ─────────────────────────────────

class TestInputValidation:
    def _samples_with(self, inputs, tmp_path, value, sample="S1", row=0, name="samples_bad.txt"):
        samples = inputs["samples"].astype(float)
        samples.loc[samples.index[row], sample] = value
        path = tmp_path / name
        samples.to_csv(path, sep="\t")
        return str(path)

    def test_negative_counts_raise(self, inputs, tmp_path):
        path = self._samples_with(inputs, tmp_path, -50)
        with pytest.raises(ValueError, match=r"^Input samples contain negative values in sample\(s\) S1 "
                                             r"\(e\.g\. S1 / A\[C>A\]A = -50\)\. Mutation counts must be "
                                             r"non-negative integers\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=path)

    def test_nan_raises(self, inputs, tmp_path):
        path = self._samples_with(inputs, tmp_path, np.nan, sample="S2")
        with pytest.raises(ValueError, match=r"^Input samples contain missing values \(NaN\) in sample\(s\) S2\. "
                                             r"Mutation counts must be finite, non-negative numbers\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=path)

    def test_inf_raises(self, inputs, tmp_path):
        path = self._samples_with(inputs, tmp_path, np.inf, sample="S3")
        with pytest.raises(ValueError, match=r"^Input samples contain infinite values in sample\(s\) S3\. "
                                             r"Mutation counts must be finite, non-negative numbers\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=path)

    PROPORTIONS_ERROR = (
        r"^All columns of the input samples sum to ~1, so the input looks like signatures or proportions rather "
        r"than mutation counts\. Activities are reported as integer mutation counts, so each column would be "
        r"treated as a single mutation\. Provide raw mutation counts; to decompose signatures into a reference set "
        r"\(COSMIC or a custom signature_database\), use decompose_fit, passing them as 'samples' and as "
        r"'signatures'\.$")

    def test_proportions_raise(self, inputs, tmp_path):
        path = tmp_path / "proportions.txt"
        (inputs["samples"] / inputs["samples"].sum()).to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=self.PROPORTIONS_ERROR):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path))

    def test_proportions_raise_in_denovo_fit(self, inputs, tmp_path):
        path = tmp_path / "proportions.txt"
        (inputs["samples"] / inputs["samples"].sum()).to_csv(path, sep="\t")
        with pytest.raises(ValueError, match=self.PROPORTIONS_ERROR):
            _run(Analyze.denovo_fit, tmp_path / "out", samples=str(path), signatures=inputs["denovo_path"])

    def test_decompose_fit_with_signatures_as_samples_warns_and_decomposes(self, inputs, tmp_path):
        # known use: decompose signatures without samples by passing them as 'samples' too
        out, printed = _run_capture(Analyze.decompose_fit, tmp_path / "out", samples=inputs["denovo_path"],
                                    signatures=inputs["denovo_path"])
        warning = ("All columns of the input samples sum to ~1, so they look like signatures or proportions "
                   "rather than mutation counts. The decomposition of 'signatures' (De_Novo_map_to_COSMIC) is not "
                   "affected, but the sample activities are rounded to integer mutation counts that add up to ~1 "
                   "per column, so they are not meaningful.")
        assert warning in printed
        assert warning in open(os.path.join(out, "JOB_METADATA_SPA.txt")).read()
        assert "non-integer values" not in printed
        table = pd.read_csv(os.path.join(out, "Decompose_Solution", "De_Novo_map_to_COSMIC_SBS96.csv"),
                            skipinitialspace=True)
        mapping = dict(zip(table.iloc[:, 0], table.iloc[:, 1]))
        assert mapping["Signature 96-SBS96A"] == "Signature SBS1 (60.00%) & Signature SBS5 (40.00%)"
        assert mapping["Signature 96-SBS96B"] == "Signature SBS13 (100.00%)"

    def test_non_integer_counts_warn_and_run(self, inputs, tmp_path):
        path = tmp_path / "fractional.txt"
        (inputs["samples"] + 0.5).to_csv(path, sep="\t")
        out, printed = _run_capture(Analyze.cosmic_fit, tmp_path / "out", samples=str(path))
        warning = ("Input samples contain non-integer values in sample(s) S1, S2, S3. Activities are reported "
                   "as integer mutation counts, so results will be rounded.")
        assert warning in printed
        assert warning in open(os.path.join(out, "JOB_METADATA_SPA.txt")).read()

    def _db_with(self, tmp_path, value, sig="SBS2"):
        db = _cosmic()[["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]].copy()
        db.loc[db.index[0], sig] = value
        path = tmp_path / "db_bad.txt"
        db.to_csv(path, sep="\t")
        return str(path)

    def test_signature_database_negative_raises(self, inputs, tmp_path):
        with pytest.raises(ValueError, match=r"^signature_database contains negative values in signature\(s\) SBS2\. "
                                             r"Signature probabilities must be non-negative\.$"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"],
                 signature_database=self._db_with(tmp_path, -0.01))

    def test_signature_database_nan_raises(self, inputs, tmp_path):
        with pytest.raises(ValueError, match=r"^signature_database contains missing values \(NaN\) in "
                                             r"signature\(s\) SBS2\."):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"],
                 signature_database=self._db_with(tmp_path, np.nan))

    def test_signature_database_inf_raises(self, inputs, tmp_path):
        with pytest.raises(ValueError, match=r"^signature_database contains infinite values in signature\(s\) SBS2\."):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"],
                 signature_database=self._db_with(tmp_path, np.inf))

    @pytest.mark.parametrize("fn", ["cosmic_fit", "decompose_fit"])
    def test_signature_database_not_summing_to_one_warns_and_runs(self, fn, inputs, tmp_path):
        db = _cosmic()[["SBS1", "SBS5", "SBS13", "SBS2", "SBS40a"]].copy()
        db["SBS2"] = db["SBS2"] * 0.85
        db["SBS13"] = db["SBS13"] * 1.2
        path = tmp_path / "db_unnormalised.txt"
        db.to_csv(path, sep="\t")
        kwargs = {"signatures": inputs["denovo_path"]} if fn == "decompose_fit" else {}
        out, printed = _run_capture(getattr(Analyze, fn), tmp_path / "out", samples=inputs["samples_path"],
                                    signature_database=str(path), **kwargs)
        warning = ("signature_database signature(s) SBS13, SBS2 do not sum to 1 (sums: 1.20, 0.85). Signatures "
                   "are expected to be probability distributions over mutation contexts.")
        assert warning in printed
        assert warning in open(os.path.join(out, "JOB_METADATA_SPA.txt")).read()


# ─── add/remove step failures ─────────────────────────────────────────────

@pytest.fixture
def add_remove_errors(monkeypatch):
    """Record exceptions raised by add_remove_signatures during a run (they are re-raised)."""
    from SigProfilerAssignment import single_sample as ss
    errors = []
    original = ss.add_remove_signatures

    def spy(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as e:
            errors.append(repr(e))
            raise

    monkeypatch.setattr(ss, "add_remove_signatures", spy)
    return errors


class TestAddRemoveStep:
    # samples that already use every available signature left nothing to add, and the add/remove
    # step failed for them; the error was swallowed and the initial composition was reported
    def test_decompose_fit_samples_using_all_decomposed_signatures(self, add_remove_errors, inputs, tmp_path):
        _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=inputs["denovo_path"])
        assert add_remove_errors == []

    def test_cosmic_fit_with_a_small_custom_database(self, add_remove_errors, inputs, tmp_path):
        db = tmp_path / "db3.txt"
        _cosmic()[["SBS1", "SBS5", "SBS13"]].to_csv(db, sep="\t")
        out = _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], signature_database=str(db))
        assert add_remove_errors == []
        _assert_close(_activities(out, "Assignment_Solution", "Assignment_Solution"), MIXTURES)

    def test_failure_is_reported_per_sample(self, monkeypatch, inputs, tmp_path):
        from SigProfilerAssignment import single_sample as ss

        def failing_step(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(ss, "add_remove_signatures", failing_step)
        out, printed = _run_capture(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"])
        metadata = open(os.path.join(out, "JOB_METADATA_SPA.txt")).read()
        for sample in ["S1", "S2", "S3"]:
            message = (
                "The add/remove step failed for sample {} (RuntimeError: injected failure); the composition "
                "from the initial removal step is reported for this sample.".format(sample)
            )
            assert message in printed
            assert message in metadata


# ─── parameters reaching the decomposition stage of decompose_fit ───────────────

@pytest.fixture
def decomposition_calls(monkeypatch):
    """Record the add/remove settings that the decomposition stage (signature_decomposition)
    passes to add_remove_signatures; the call itself still runs."""
    import inspect
    from SigProfilerAssignment import single_sample as ss
    calls = []
    original = ss.add_remove_signatures

    def spy(*args, **kwargs):
        if inspect.stack()[1].function == "signature_decomposition":
            calls.append({k: kwargs.get(k) for k in ("add_penalty", "remove_penalty", "connected_sigs")})
        return original(*args, **kwargs)

    monkeypatch.setattr(ss, "add_remove_signatures", spy)
    return calls


class TestDecompositionParameters:
    def test_defaults_reach_the_decomposition(self, decomposition_calls, inputs, tmp_path):
        _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=inputs["denovo_path"])
        assert decomposition_calls
        assert all(c["add_penalty"] == 0.05 and c["remove_penalty"] == 0.01 for c in decomposition_calls)

    def test_user_penalties_reach_the_decomposition(self, decomposition_calls, inputs, tmp_path):
        # the decomposition used fixed 0.05 / 0.01
        _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=inputs["denovo_path"],
             nnls_add_penalty=0.2, nnls_remove_penalty=0.1)
        assert decomposition_calls
        assert all(c["add_penalty"] == 0.2 and c["remove_penalty"] == 0.1 for c in decomposition_calls)

    def test_default_connected_sigs_reaches_the_decomposition(self, decomposition_calls, inputs, tmp_path):
        _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=inputs["denovo_path"])
        assert decomposition_calls
        assert all(c["connected_sigs"] is True for c in decomposition_calls)

    def test_connected_sigs_false_reaches_the_decomposition(self, decomposition_calls, inputs, tmp_path):
        # getProcessAvg's default True overwrote the user's False
        _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=inputs["denovo_path"],
             connected_sigs=False)
        assert decomposition_calls
        assert all(c["connected_sigs"] is False for c in decomposition_calls)


# ─── input errors ──────────────────────────────────────────────

class TestInputErrors:
    def test_missing_samples_file_raises_file_not_found(self, tmp_path):
        # a mistyped path used to fail later with
        # "AttributeError: 'str' object has no attribute 'shape'"
        missing = tmp_path / "typo_samples.txt"
        output = tmp_path / "out"
        with pytest.raises(FileNotFoundError, match="typo_samples.txt"):
            _run(Analyze.cosmic_fit, output, samples=str(missing))
        assert not output.exists()

    # library errors must raise exceptions, never sys.exit()
    def test_invalid_input_type_raises_value_error(self, inputs, tmp_path):
        with pytest.raises(ValueError, match="Invalid input_type"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], input_type="bam")

    def test_exclude_signature_subgroups_not_a_list_raises_value_error(self, inputs, tmp_path):
        with pytest.raises(ValueError, match="exclude_signature_subgroups"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"],
                 exclude_signature_subgroups="UV_signatures")

    @pytest.mark.parametrize("fn", [Analyze.denovo_fit, Analyze.decompose_fit])
    def test_missing_signatures_file_raises_file_not_found(self, fn, inputs, tmp_path):
        with pytest.raises(FileNotFoundError, match="typo_signatures.txt"):
            _run(fn, tmp_path / "out", samples=inputs["samples_path"],
                 signatures=str(tmp_path / "typo_signatures.txt"))

    def test_missing_signature_database_raises_file_not_found(self, inputs, tmp_path):
        with pytest.raises(FileNotFoundError, match="typo_db.txt"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"],
                 signature_database=str(tmp_path / "typo_db.txt"))

    def test_missing_signature_database_in_decompose_fit_raises_file_not_found(self, inputs, tmp_path):
        with pytest.raises(FileNotFoundError, match="typo_db.txt"):
            _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"],
                 signatures=inputs["denovo_path"], signature_database=str(tmp_path / "typo_db.txt"))

    def test_signature_database_with_other_row_count_raises_value_error(self, inputs, tmp_path):
        db = pd.DataFrame(np.full((10, 2), 0.1), index=["CTX%02d" % i for i in range(10)], columns=["A", "B"])
        db_path = tmp_path / "db10.txt"
        db.to_csv(db_path, sep="\t")
        with pytest.raises(ValueError, match="Input samples have 96 rows but signature_database has 10"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=inputs["samples_path"], signature_database=str(db_path))

    # without a signature_database, an unsupported context must not
    # silently use the samples themselves as the signature database
    @pytest.mark.parametrize("collapse", [False, True])
    def test_unsupported_context_without_database_raises_value_error(self, inputs, tmp_path, collapse):
        samples95 = inputs["samples"].drop(inputs["samples"].index[10])
        path = tmp_path / "samples95.txt"
        samples95.to_csv(path, sep="\t")
        with pytest.raises(ValueError, match="Input samples have 95 rows .* no COSMIC reference"):
            _run(Analyze.cosmic_fit, tmp_path / "out", samples=str(path), collapse_to_SBS96=collapse)

    def test_decompose_unsupported_signature_context_without_database_raises_value_error(self, tmp_path):
        samples, db, _ = _custom_context_inputs(tmp_path, ["CTX%02d" % i for i in range(10)])
        with pytest.raises(ValueError, match="Input samples have 10 rows .* no COSMIC reference"):
            _run(Analyze.decompose_fit, tmp_path / "out", samples=samples, signatures=db,
                 collapse_to_SBS96=False)

    def test_decompose_signatures_with_other_row_count_than_samples_raises_value_error(self, inputs, tmp_path):
        # de novo signatures are aligned to the sample contexts by position
        denovo = _dummy_denovo_signatures()
        path = tmp_path / "denovo95.txt"
        denovo.drop(denovo.index[10]).to_csv(path, sep="\t")
        with pytest.raises(ValueError, match="The matrix passed in 'signatures' has 95 rows but the samples have 96"):
            _run(Analyze.decompose_fit, tmp_path / "out", samples=inputs["samples_path"], signatures=str(path))



# ─── denovo_fit ──────────────────────────────────────────────

class TestDenovoFit:
    EXPECTED = {
        "S1": {"SBS96A": 2704, "SBS96B": 796},
        "S2": {"SBS96A": 1801},
        "S3": {"SBS96A": 1373, "SBS96B": 1229},
    }

    def test_activities_on_dummy_signatures(self, denovo_run):
        acts = _activities(denovo_run, "De_Novo_Solution", "De_Novo")
        _assert_close(acts, self.EXPECTED)

    def test_activities_conserve_mutation_burden(self, denovo_run, inputs):
        acts = _activities(denovo_run, "De_Novo_Solution", "De_Novo")
        burden = inputs["samples"].sum()
        assert (acts.sum(axis=1) == burden.loc[acts.index]).all()

    def test_writes_expected_files(self, denovo_run):
        for rel in (
            "JOB_METADATA_SPA.txt",
            "De_Novo_Solution/Activities/De_Novo_MutationType_Probabilities.txt",
            "De_Novo_Solution/Signatures/De_Novo_Signatures.txt",
            "De_Novo_Solution/Solution_Stats/De_Novo_Samples_Stats.txt",
        ):
            assert os.path.isfile(os.path.join(denovo_run, rel)), rel


# ─── decompose_fit ──────────────────────────────────────────────

class TestDecomposeFit:
    def test_decomposes_dummy_denovo_signatures(self, decompose_run):
        path = os.path.join(decompose_run, "Decompose_Solution", "De_Novo_map_to_COSMIC_SBS96.csv")
        table = pd.read_csv(path, skipinitialspace=True)
        mapping = dict(zip(table.iloc[:, 0], table.iloc[:, 1]))
        assert mapping["Signature 96-SBS96A"] == "Signature SBS1 (60.00%) & Signature SBS5 (40.00%)"
        assert mapping["Signature 96-SBS96B"] == "Signature SBS13 (100.00%)"

    def test_recovers_dummy_mixtures(self, decompose_run):
        acts = _activities(decompose_run, "Decompose_Solution", "Decompose_Solution")
        _assert_close(acts, MIXTURES)

    def test_activities_conserve_mutation_burden(self, decompose_run, inputs):
        acts = _activities(decompose_run, "Decompose_Solution", "Decompose_Solution")
        burden = inputs["samples"].sum()
        assert (acts.sum(axis=1) == burden.loc[acts.index]).all()
