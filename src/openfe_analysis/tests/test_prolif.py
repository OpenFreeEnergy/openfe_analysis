import MDAnalysis as mda
import numpy as np
import prolif as plf
import pytest
from rdkit.Chem import Lipinski

from openfe_analysis.reader import FEReader
from openfe_analysis.prolif import ProLIFAnalysis


@pytest.fixture
def universe(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """Skipped-simulation Universe (FEReader) for the ProLIF tests."""
    return mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )


@pytest.fixture
def ligand_ag(universe):
    """The ligand AtomGroup (resname UNK)."""
    return universe.select_atoms("resname UNK")


@pytest.fixture
def make_analysis():
    """Factory: a bare ProLIFAnalysis wired with dummy universe/ligand/fp."""

    def _make(trajectory):
        analysis = object.__new__(ProLIFAnalysis)
        analysis.universe = type("U", (), {"trajectory": trajectory})()
        analysis.ligand_ag = object()
        analysis.protein_ag = object()
        analysis.fp = type("FP", (), {"run": lambda self, *a, **k: None})()
        return analysis

    return _make


def test_prolifanalysis_runs_vdwcontact(universe, ligand_ag):
    """
    Test for identification of interactions
    """
    analysis = ProLIFAnalysis(
        universe, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )
    analysis.run(stop=5, step=1, n_jobs=1, progress=False)

    df = analysis.to_dataframe(dtype=np.uint8)
    assert df.shape[0] == 5
    # only VdWContact was requested
    assert set(df.columns.get_level_values("interaction")) == {"VdWContact"}
    assert hasattr(analysis.fp, "ifp")
    assert len(analysis.fp.ifp) == 5

    assert analysis.ifp is analysis.fp.ifp
    assert len(analysis.ifp) == 5

    # Ensure there is at least one detected interaction across all processed frames
    assert sum(len(v) for v in analysis.fp.ifp.values()) > 0


def test_run_slice_sets_frames_times_nframes(universe, ligand_ag):
    """run() slicing should set frames, n_frames and times consistently."""
    analysis = ProLIFAnalysis(universe, ligand_ag, interactions=["VdWContact"])
    analysis.run(start=0, stop=5, step=2, n_jobs=1, progress=False)

    assert list(analysis.frames) == [0, 2, 4]
    assert analysis.n_frames == 3
    assert analysis.times is not None
    assert len(analysis.times) == 3
    np.testing.assert_allclose(
        analysis.times, analysis.frames * universe.trajectory.dt
    )


def test_run_frame_metadata_reset_to_none_on_error(make_analysis):
    """run() resets frame metadata to None on error."""

    class FailingTraj:
        # len() raising is what trips the metadata fallback in run()
        def __getitem__(self, s):
            return "sliced"

        def __len__(self):
            raise RuntimeError("boom")

    analysis = make_analysis(FailingTraj())
    result = analysis.run(n_jobs=1, progress=False)

    assert result is analysis
    assert analysis.frames is None
    assert analysis.times is None
    assert analysis.n_frames is None


def test_guess_bonds_enables_protein_chemistry(universe, ligand_ag):
    """
    Test for protein connectivity
    """
    analysis = ProLIFAnalysis(
        universe, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )

    # pick a residue from the pocket and check it has connectivity in RDKit
    universe.trajectory[0]
    res_atoms = analysis.protein_ag.residues[0].atoms
    res_mol = res_atoms.convert_to("RDKIT", implicit_hydrogens=False)
    assert res_mol.GetNumBonds() > 0

    # ensure the protein donors/acceptors exist
    prot_mol = analysis.protein_ag.convert_to("RDKIT", implicit_hydrogens=False)
    assert Lipinski.NumHDonors(prot_mol) + Lipinski.NumHAcceptors(prot_mol) > 0


def test_prolifanalysis_accepts_all_keyword(universe, ligand_ag):
    """
    The string "all" should track every available ProLIF interaction
    (bridged included), except WaterBridge which is dropped when the
    system has no water.
    """
    with pytest.warns(UserWarning, match="WaterBridge selected"):
        analysis = ProLIFAnalysis(
            universe, ligand_ag, interactions="all", guess_bonds=True
        )

    available = set(plf.Fingerprint.list_available(show_bridged=True))
    # this test system has no water, so WaterBridge is dropped
    assert set(analysis.fp.interactions) == available - {"WaterBridge"}


def test_default_interactions_are_prolif_defaults(universe, ligand_ag):
    """interactions=None should track ProLIF's DEFAULT_INTERACTIONS."""
    analysis = ProLIFAnalysis(universe, ligand_ag, interactions=None)

    expected = {
        "Hydrophobic",
        "HBDonor",
        "HBAcceptor",
        "PiStacking",
        "Anionic",
        "Cationic",
        "CationPi",
        "PiCation",
        "VdWContact",
    }
    assert set(analysis.fp.interactions) == expected


def test_waterbridge_empty_selection_warns_and_raises(universe, ligand_ag, monkeypatch):
    """
    Selecting only WaterBridge in a water-free system should warn and then
    raise an error about missing interactions.
    """
    original_select_atoms = universe.select_atoms

    def patched_select_atoms(selection, *args, **kwargs):
        if selection == "water":
            return universe.atoms[[]]
        return original_select_atoms(selection, *args, **kwargs)

    monkeypatch.setattr(universe, "select_atoms", patched_select_atoms)

    with pytest.warns(UserWarning, match="WaterBridge selected"):
        with pytest.raises(ValueError, match="No interactions left"):
            ProLIFAnalysis(
                universe,
                ligand_ag,
                interactions=["WaterBridge"],
                guess_bonds=True,
            )


def test_guess_bonds_false_skips_guessing(universe, ligand_ag):
    """guess_bonds=False should not add bonds (leaves the ligand untouched)."""
    n_bonds_before = len(ligand_ag.bonds)

    analysis = ProLIFAnalysis(
        universe, ligand_ag, interactions=["VdWContact"], guess_bonds=False
    )

    assert analysis.fp is not None
    # guessing was skipped, so no bonds were added to the ligand
    assert len(analysis.ligand_ag.bonds) == n_bonds_before
