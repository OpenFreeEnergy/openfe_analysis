import MDAnalysis as mda
import numpy as np
import pytest
from rdkit.Chem import Lipinski

from openfe_analysis.reader import FEReader
from openfe_analysis.prolif import ProLIFAnalysis


def test_prolifanalysis_runs_vdwcontact(
    simulation_skipped_nc, hybrid_system_skipped_pdb
):
    """
    Test for identification of interactions
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(
        u, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )
    analysis.run(stop=5, step=1, n_jobs=1, progress=False)

    df = analysis.to_dataframe(dtype=np.uint8)
    assert df.shape[0] == 5
    assert hasattr(analysis.fp, "ifp")
    assert len(analysis.fp.ifp) == 5

    # Check AnalysisBase
    assert hasattr(analysis, "results")
    assert hasattr(analysis.results, "ifp")
    assert analysis.results.ifp is analysis.fp.ifp
    assert len(analysis.results.ifp) == 5

    assert analysis.results.ifp_df is df

    # Ensure there is at least one detected interaction across all processed frames
    assert sum(len(v) for v in analysis.fp.ifp.values()) > 0


def test_guess_bonds_enables_protein_chemistry(
    simulation_skipped_nc, hybrid_system_skipped_pdb
):
    """
    Test for protein connectivity
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(
        u, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )

    # pick a residue from the pocket and check it has connectivity in RDKit
    u.trajectory[0]
    res_atoms = analysis.protein_ag.residues[0].atoms
    res_mol = res_atoms.convert_to("RDKIT", implicit_hydrogens=False)
    assert res_mol.GetNumBonds() > 0

    # ensure the protein donors/acceptors exist
    prot_mol = analysis.protein_ag.convert_to("RDKIT", implicit_hydrogens=False)
    assert Lipinski.NumHDonors(prot_mol) + Lipinski.NumHAcceptors(prot_mol) > 0


def test_prolifanalysis_accepts_all_keyword(
    simulation_skipped_nc, hybrid_system_skipped_pdb
):
    """
    The string "all" should be accepted as the special keyword for
    all available ProLIF interactions.
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(u, ligand_ag, interactions="all", guess_bonds=True)

    assert analysis.fp is not None


def test_waterbridge_empty_selection_warns_and_skips_parameters(
    simulation_skipped_nc, hybrid_system_skipped_pdb, monkeypatch
):
    """
    Requesting WaterBridge with an empty water selection should warn
    instead of raising, and should not configure WaterBridge parameters.
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    original_select_atoms = u.select_atoms

    def patched_select_atoms(selection, *args, **kwargs):
        if selection == "water and byres around 8 (group ligand or group pocket)":
            return u.atoms[[]]
        return original_select_atoms(selection, *args, **kwargs)

    monkeypatch.setattr(u, "select_atoms", patched_select_atoms)

    with pytest.warns(UserWarning, match="WaterBridge selected"):
        analysis = ProLIFAnalysis(
            u,
            ligand_ag,
            interactions=["WaterBridge"],
            guess_bonds=True,
        )

    assert analysis._parameters is None


def test_plot_2d_builds_ligand_mol_and_delegates(
    simulation_skipped_nc, hybrid_system_skipped_pdb, monkeypatch
):
    """
    plot_2d should build a ligand molecule internally when one is not
    provided and delegate to ProLIF's plot_lignetwork.
    """
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, index=0
    )
    ligand_ag = u.select_atoms("resname UNK")

    analysis = ProLIFAnalysis(
        u, ligand_ag, interactions=["VdWContact"], guess_bonds=True
    )

    analysis.fp.ifp = {0: {"dummy": []}}

    fake_ligand_mol = object()
    calls = {}

    def fake_from_mda(atomgroup, **kwargs):
        calls["from_mda"] = (atomgroup, kwargs)
        return fake_ligand_mol

    def fake_plot_lignetwork(ligand_mol, **kwargs):
        calls["plot_lignetwork"] = (ligand_mol, kwargs)
        return "fake-view"

    monkeypatch.setattr(
        "openfe_analysis.prolif.plf.Molecule.from_mda",
        fake_from_mda,
    )
    monkeypatch.setattr(analysis.fp, "plot_lignetwork", fake_plot_lignetwork)

    view = analysis.plot_2d(frame=0, kind="frame")

    assert view == "fake-view"
    assert calls["from_mda"][0] is ligand_ag
    assert calls["from_mda"][1]["inferrer"] is None
    assert calls["from_mda"][1]["implicit_hydrogens"] is False
    assert calls["from_mda"][1]["use_segid"] == analysis.fp.use_segid
    assert calls["plot_lignetwork"][0] is fake_ligand_mol
    assert calls["plot_lignetwork"][1]["frame"] == 0
    assert calls["plot_lignetwork"][1]["kind"] == "frame"