import pytest
import argparse
import os
import SigProfilerAssignment
from SigProfilerAssignment.controllers import cli_controller
from SigProfilerAssignment.controllers.cli_controller import (
    parse_arguments_common,
    str2bool,
    str2list
)


def test_default_values():
    args = parse_arguments_common(
        ["dummy_sample", "dummy_output"], "Test default values"
    )
    assert args.make_plots == True
    assert args.collapse_to_SBS96 == True
    assert args.connected_sigs == True
    assert args.verbose == False
    assert args.export_probabilities == True
    assert args.export_probabilities_per_mutation == False
    assert args.exome == False


def test_argument_parsing():
    args = parse_arguments_common(
        [
            "dummy_sample",
            "dummy_output",
            "--genome_build",
            "GRCh38",
            "--cosmic_version",
            "3.4",
            "--make_plots",
            "False",
            "--collapse_to_SBS96",
            "False",
            "--connected_sigs",
            "False",
            "--verbose",
            "True",
            "--export_probabilities",
            "False",
            "--export_probabilities_per_mutation",
            "True",
            "--exome",
            "True",
            "--sample_reconstruction_plots",
            "png",
            "--exclude_signature_subgroups",
            "MMR_deficiency_signatures,POL_deficiency_signatures,HR_deficiency_signatures,BER_deficiency_signatures,Chemotherapy_signatures,Immunosuppressants_signatures,Treatment_signatures,APOBEC_signatures,Tobacco_signatures,UV_signatures,AA_signatures,Colibactin_signatures,Artifact_signatures,Lymphoid_signatures"
        ],
        "Test argument parsing",
    )
    assert args.samples == "dummy_sample"
    assert args.output == "dummy_output"
    assert args.genome_build == "GRCh38"
    assert args.cosmic_version == 3.4
    assert args.make_plots == False
    assert args.collapse_to_SBS96 == False
    assert args.connected_sigs == False
    assert args.verbose == True
    assert args.export_probabilities == False
    assert args.export_probabilities_per_mutation == True
    assert args.exome == True
    assert args.sample_reconstruction_plots == "png"
    assert args.exclude_signature_subgroups == ['MMR_deficiency_signatures', 'POL_deficiency_signatures', 'HR_deficiency_signatures' , 'BER_deficiency_signatures', 'Chemotherapy_signatures', 'Immunosuppressants_signatures', 'Treatment_signatures', 'APOBEC_signatures', 'Tobacco_signatures', 'UV_signatures', 'AA_signatures', 'Colibactin_signatures', 'Artifact_signatures', 'Lymphoid_signatures']


def test_boolean_conversion():
    assert str2bool("yes") == True
    assert str2bool("true") == True
    assert str2bool("t") == True
    assert str2bool("y") == True
    assert str2bool("1") == True
    assert str2bool("no") == False
    assert str2bool("false") == False
    assert str2bool("f") == False
    assert str2bool("n") == False
    assert str2bool("0") == False
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")

def test_str2list():
    assert str2list("arg1,arg2,arg3") == ["arg1", "arg2","arg3"]
    assert str2list("arg_unique") == ["arg_unique"]
    assert str2list("wrong.sepparator") == ["wrong.sepparator"]


# ─── CliController dispatch ──────────────────────────────────────────────


@pytest.fixture
def captured_call(monkeypatch):
    """Replace spa_analyze with a recorder so dispatch can be tested without running an analysis."""
    calls = []
    monkeypatch.setattr(cli_controller.decomp, "spa_analyze", lambda **kw: calls.append(kw))
    return calls


class TestDispatch:
    COMMON = ["in.txt", "out_dir", "--genome_build", "GRCh38", "--cosmic_version", "3.4",
              "--nnls_add_penalty", "0.1", "--nnls_remove_penalty", "0.02",
              "--connected_sigs", "False", "--cpu", "2"]

    def _check_common(self, kw):
        assert kw["samples"] == "in.txt"
        assert kw["output"] == "out_dir"
        assert kw["genome_build"] == "GRCh38"
        assert kw["cosmic_version"] == 3.4
        assert kw["nnls_add_penalty"] == 0.1
        assert kw["nnls_remove_penalty"] == 0.02
        assert kw["connected_sigs"] is False
        assert kw["cpu"] == 2

    def test_cosmic_fit(self, captured_call):
        cli_controller.CliController().dispatch_cosmic_fit(self.COMMON)
        (kw,) = captured_call
        self._check_common(kw)
        assert kw["cosmic_fit_option"] is True
        assert kw["decompose_fit_option"] is False
        assert kw["denovo_refit_option"] is False

    def test_denovo_fit(self, captured_call):
        cli_controller.CliController().dispatch_denovo_fit(self.COMMON + ["--signatures", "sigs.txt"])
        (kw,) = captured_call
        self._check_common(kw)
        assert kw["signatures"] == "sigs.txt"
        assert kw["denovo_refit_option"] is True
        assert kw["cosmic_fit_option"] is False

    def test_decompose_fit(self, captured_call):
        cli_controller.CliController().dispatch_decompose_fit(self.COMMON + ["--signatures", "sigs.txt"])
        (kw,) = captured_call
        self._check_common(kw)
        assert kw["signatures"] == "sigs.txt"
        assert kw["decompose_fit_option"] is True
        assert kw["cosmic_fit_option"] is False

if __name__ == "__main__":
    pytest.main()


class TestCosmicVersionArgument:
    # the version is used to build the reference file name (e.g. COSMIC_v3_SBS_GRCh37.txt)
    @pytest.mark.parametrize("value,name", [("1", "1"), ("2", "2"), ("3", "3"), ("3.0", "3"),
                                            ("3.1", "3.1"), ("3.4", "3.4"), ("3.6", "3.6")])
    def test_version_keeps_the_reference_file_name(self, value, name):
        args = parse_arguments_common(["in.txt", "out_dir", "--cosmic_version", value], "test")
        assert str(args.cosmic_version) == name
        ref = os.path.join(os.path.dirname(SigProfilerAssignment.__file__), "data", "Reference_Signatures",
                           "GRCh37", "COSMIC_v{}_SBS_GRCh37.txt".format(args.cosmic_version))
        assert os.path.exists(ref)

    def test_default_version(self):
        args = parse_arguments_common(["in.txt", "out_dir"], "test")
        assert str(args.cosmic_version) == "3.6"
