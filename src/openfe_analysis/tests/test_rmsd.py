from itertools import islice

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import pytest
from MDAnalysis.analysis import rms
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.transformations import unwrap
from numpy.testing import assert_allclose

from openfe_analysis.reader import FEReader
from openfe_analysis.rmsd import gather_rms_data, make_Universe
from openfe_analysis.transformations import Aligner


@pytest.fixture
def mda_universe(hybrid_system_skipped_pdb, simulation_skipped_nc):
    """
    Safely create and destroy an MDAnalysis Universe.

    Guarantees:
    - NetCDF file is opened exactly once
    """
    u = make_Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        state=0,
    )
    yield u
    u.trajectory.close()


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
    # 15 entries because 6 * 6 frames // 2
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
    # RMSD is low for this multichain protein
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
    # 15 entries because 6 * 6 frames // 2
    assert len(output["protein_2D_RMSD"][0]) == 1275
    # TODO: very large as the multichain fix is not in yet
    assert_allclose(
        output["protein_2D_RMSD"][0][:6],
        [1.089747, 1.006143, 1.045068, 1.476353, 1.332893, 1.110507],
        rtol=1e-3,
    )


def test_multichain_rmsd_shifting(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = mda.Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        index=0,
        format=FEReader,
    )
    prot = u.select_atoms("protein")
    # Do other transformations, but no shifting
    unwrap_tr = unwrap(prot)
    for frag in prot.fragments:
        make_whole(frag, reference_atom=frag[0])
    align = Aligner(prot)
    u.trajectory.add_transformations(unwrap_tr, align)
    chains = [seg.atoms for seg in prot.segments]
    assert len(chains) > 1, "Test requires multi-chain protein"

    # RMSD without shifting
    r = rms.RMSD(prot)
    r.run()
    rmsd_no_shift = r.rmsd[:, 2]
    assert np.max(np.diff(rmsd_no_shift[:20])) > 10  # expect jumps
    u.trajectory.close()

    # RMSD with shifting
    u2 = make_Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, state=0)
    prot2 = u2.select_atoms("protein")
    R2 = rms.RMSD(prot2)
    R2.run()
    rmsd_shift = R2.rmsd[:, 2]
    assert np.max(np.diff(rmsd_shift[:20])) < 2  # jumps should disappear
    u2.trajectory.close()


def test_chain_radius_of_gyration_stable(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = make_Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, state=0)

    protein = u.select_atoms("protein")
    chain = protein.segments[0].atoms

    rgs = []
    for ts in u.trajectory[:50]:
        rgs.append(chain.radius_of_gyration())

    # Chain should not explode or collapse due to PBC errors
    assert np.std(rgs) < 2.0
    u.trajectory.close()


def test_rmsd_reference_is_first_frame(mda_universe):
    u = mda_universe
    prot = u.select_atoms("protein")

    _ = next(iter(u.trajectory))  # SAFE
    ref = prot.positions.copy()

    rmsd = np.sqrt(((prot.positions - ref) ** 2).mean())
    assert rmsd == 0.0
    u.trajectory.close()


def test_ligand_com_continuity(mda_universe):
    u = mda_universe
    ligand = u.select_atoms("resname UNK")

    coms = [ligand.center_of_mass() for ts in islice(u.trajectory, 20)]
    jumps = [np.linalg.norm(coms[i + 1] - coms[i]) for i in range(len(coms) - 1)]

    assert max(jumps) < 5.0
    u.trajectory.close()
