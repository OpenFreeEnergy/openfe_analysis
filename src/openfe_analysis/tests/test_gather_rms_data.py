import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose

from openfe_analysis.rmsd import _select_ligand, gather_rms_data


def test_gather_rms_data_regression(simulation_nc, hybrid_system_pdb):
    output = gather_rms_data(
        hybrid_system_pdb,
        simulation_nc,
        skip=100,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 3
    assert_allclose(
        output["protein_RMSD"][0],
        [0.0, 1.003, 1.276, 1.263, 1.516, 1.251],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 3
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 0.9094, 1.0398, 0.9774, 1.9108, 1.2149],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 3
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.5458, 0.8364, 0.4914, 1.1939, 0.7587],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 3
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.0029, 1.2756, 1.2635, 1.5165, 1.2509, 1.0882],
        rtol=1e-3,
    )


def test_gather_rms_data_regression_skippednc(simulation_skipped_nc, hybrid_system_skipped_pdb):
    output = gather_rms_data(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        skip=None,
    )

    assert_allclose(output["time(ps)"], np.arange(0, 5001, 100))
    assert len(output["protein_RMSD"]) == 11
    assert_allclose(
        output["protein_RMSD"][0][:6],
        [0, 1.089747, 1.006143, 1.045068, 1.476353, 1.332893],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 11
    assert_allclose(
        output["ligand_RMSD"][0][:6],
        [0.0, 1.092039, 0.839234, 1.228383, 1.533331, 1.276798],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 11
    assert_allclose(
        output["ligand_wander"][0][:6],
        [0.0, 0.908097, 0.674262, 0.971328, 0.909263, 1.101882],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 11
    assert len(output["protein_2D_RMSD"][0]) == 1275
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.089747, 1.006143, 1.045068, 1.476353, 1.332893, 1.110507],
        rtol=1e-3,
    )


def test_gather_rms_data_ligand_only(simulation_skipped_nc, hybrid_system_skipped_pdb):
    output = gather_rms_data(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        skip=100,
        protein_selection="resname DOESNOTEXIST",
    )

    assert len(output["protein_RMSD"]) == 0
    assert len(output["protein_2D_RMSD"]) == 0
    assert len(output["ligand_RMSD"]) > 0
    assert len(output["ligand_wander"]) > 0


def _make_test_universe(tempfactors, resids, resnames=None):
    n_atoms = len(tempfactors)
    n_residues = len(set(resids))
    unique_resids = list(dict.fromkeys(resids))
    resid_to_idx = {r: i for i, r in enumerate(unique_resids)}
    atom_resindex = [resid_to_idx[r] for r in resids]

    if resnames is None:
        resnames = ["UNK"] * n_residues

    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindex,
    )
    u.add_TopologyAttr("names", [f"C{i}" for i in range(1, n_atoms + 1)])
    u.add_TopologyAttr("resnames", resnames)
    u.add_TopologyAttr("resids", unique_resids)
    u.add_TopologyAttr("tempfactors", tempfactors)
    return u


def test_select_ligand_single_unk():
    u = _make_test_universe(
        tempfactors=[0.25, 0.50, 0.75],
        resids=[1, 1, 1],
    )
    ligand = _select_ligand(u, "resname UNK")
    assert len(ligand) == 3
    assert set(ligand.resids) == {1}


def test_select_ligand_cofactor_present():
    u = _make_test_universe(
        tempfactors=[0.25, 0.50, 0.75, 0.00],
        resids=[1, 1, 1, 2],
    )
    ligand = _select_ligand(u, "resname UNK")
    assert len(ligand) == 3
    assert set(ligand.resids) == {1}


def test_select_ligand_multiple_hybrid():
    u = _make_test_universe(
        tempfactors=[0.25, 0.50, 0.75, 0.25, 0.50],
        resids=[1, 1, 1, 2, 2],
    )
    ligand = _select_ligand(u, "resname UNK")
    assert len(ligand) == 3
    assert set(ligand.resids) == {1}


def test_select_ligand_no_hybrid_fallback():
    u = _make_test_universe(
        tempfactors=[0.00, 0.00, 0.00, 0.00],
        resids=[1, 1, 2, 2],
    )
    ligand = _select_ligand(u, "resname UNK")
    assert len(ligand) == 2


def test_select_ligand_custom_selection_unchanged():
    u = _make_test_universe(
        tempfactors=[0.25, 0.50, 0.75, 0.00],
        resids=[1, 1, 1, 2],
        resnames=["LIG", "LIG", "LIG", "COF"],
    )
    ligand = _select_ligand(u, "resname LIG")
    assert len(ligand) == 3
    assert set(ligand.resids) == {1}
