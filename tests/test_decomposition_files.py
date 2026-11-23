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


# ─── denovo_fit ──────────────────────────────────────────────

class TestDenovoFit:
    EXPECTED = {
        "S1": {"SBS96A": 2704, "SBS96B": 796},
        "S2": {"SBS96A": 1682, "SBS96B": 119},
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
