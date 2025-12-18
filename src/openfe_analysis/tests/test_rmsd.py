import netCDF4 as nc
import numpy as np
import pytest
from numpy.testing import assert_allclose
from MDAnalysis.analysis import rms
from openfe_analysis.rmsd import gather_rms_data, make_Universe


@pytest.mark.flaky(reruns=3)
def test_gather_rms_data_regression(simulation_nc, hybrid_system_pdb):
    output = gather_rms_data(
        hybrid_system_pdb,
        simulation_nc,
        skip=100,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 11
    assert_allclose(
        output["protein_RMSD"][0],
        [0.0, 1.088, 1.009, 1.120, 1.026, 1.167],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 11
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 0.9434, 0.8068, 0.8255, 1.2313, 0.7186],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 11
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.8128, 0.5010, 0.6392, 1.1071, 0.3021],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 11
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.0884, 1.0099, 1.1200, 1.0267, 1.1673, 1.2378],
        rtol=1e-3,
    )


@pytest.mark.flaky(reruns=3)
def test_gather_rms_data_regression_skippednc(simulation_skipped_nc, hybrid_system_skipped_pdb):
    output = gather_rms_data(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        skip=None,
    )

    assert_allclose(output["time(ps)"], [0.0, 100.0, 200.0, 300.0, 400.0, 500.0])
    assert len(output["protein_RMSD"]) == 11
    assert_allclose(
        output["protein_RMSD"][0],
        [0, 1.176307, 1.203364, 1.486987, 1.17462, 1.143457],
        rtol=1e-3,
    )
    assert len(output["ligand_RMSD"]) == 11
    assert_allclose(
        output["ligand_RMSD"][0],
        [0.0, 1.066418, 1.314562, 1.051574, 0.451605, 0.706698],
        rtol=1e-3,
    )
    assert len(output["ligand_wander"]) == 11
    assert_allclose(
        output["ligand_wander"][0],
        [0.0, 0.726258, 0.628337, 0.707796, 0.329651, 0.483037],
        rtol=1e-3,
    )
    assert len(output["protein_2D_RMSD"]) == 11
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 15
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.176307, 1.203364, 1.486987, 1.17462, 1.143457, 1.244173],
        rtol=1e-3,
    )

def test_multichain_com_continuity(simulation_nc_multichain, system_pdb_multichain):
    u = make_Universe(system_pdb_multichain, simulation_nc_multichain, state=0)
    prot = u.select_atoms("protein")
    chains = [seg.atoms for seg in prot.segments]
    assert len(chains) == 2

    segments = prot.segments
    assert len(segments) > 1, "Test requires multi-chain protein"

    chain_a = segments[0].atoms
    chain_b = segments[1].atoms

    distances = []
    for ts in u.trajectory[:50]:
        d = np.linalg.norm(
            chain_a.center_of_mass() - chain_b.center_of_mass()
        )
        distances.append(d)

    # No large frame-to-frame jumps (PBC artifacts)
    jumps = np.abs(np.diff(distances))
    assert np.max(jumps) < 5.0  # Å
    del u

# def test_chain_radius_of_gyration_stable(simulation_nc_multichain, system_pdb_multichain):
#     u = make_Universe(system_pdb_multichain, simulation_nc_multichain, state=0)
#
#     protein = u.select_atoms("protein")
#     chain = protein.segments[0].atoms
#
#     rgs = []
#     for ts in u.trajectory[:50]:
#         rgs.append(chain.radius_of_gyration())
#
#     # Chain should not explode or collapse due to PBC errors
#     assert np.std(rgs) < 2.0

def test_rmsd_continuity(simulation_nc_multichain, system_pdb_multichain):
    u = make_Universe(system_pdb_multichain, simulation_nc_multichain, state=0)

    prot = u.select_atoms("protein and name CA")
    ref = prot.positions.copy()

    rmsds = []
    for ts in u.trajectory[:20]:
        diff = prot.positions - ref
        rmsd = np.sqrt((diff * diff).sum(axis=1).mean())
        rmsds.append(rmsd)

    jumps = np.abs(np.diff(rmsds))
    assert np.max(jumps) < 2.0
    del u


def test_rmsd_reference_is_first_frame(simulation_nc_multichain, system_pdb_multichain):
    # RMS of first frame should be zero
    u = make_Universe(system_pdb_multichain, simulation_nc_multichain, state=0)
    prot = u.select_atoms("protein")

    u.trajectory[0]
    ref = prot.positions.copy()

    u.trajectory[0]
    r0 = rms.rmsd(prot.positions, ref, center=False, superposition=False)

    assert r0 == pytest.approx(0.0)
    del u

def test_ligand_com_continuity(simulation_nc_multichain, system_pdb_multichain):
    u = make_Universe(system_pdb_multichain, simulation_nc_multichain, state=0)
    ligand = u.select_atoms("resname UNK")

    coms = [ligand.center_of_mass() for ts in u.trajectory[:50]]
    jumps = [np.linalg.norm(coms[i+1] - coms[i]) for i in range(len(coms)-1)]

    assert max(jumps) < 5.0
    del u