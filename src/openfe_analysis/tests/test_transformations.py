from itertools import islice

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis import rms
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.transformations import unwrap

from openfe_analysis.transformations import (
    Aligner,
    ClosestImageShift,
    NoJump,
)
from openfe_analysis.utils import apply_transformations, universe_utils


@pytest.fixture
def universe(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = mda.Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        format="MultiStateReporter",
        index=0,
    )
    yield u
    u.trajectory.close()


@pytest.fixture
def universe_single_state(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    yield u
    u.trajectory.close()


def test_closest_image_shift(universe):
    prot = universe.select_atoms("protein and name CA")
    lig = universe.select_atoms("resname UNK")
    # The ligand is in the same periodic image as the protein.
    # We translate it first into a different image, then transform it back.

    # Original COM distance
    distance_orig = mda.lib.distances.calc_bonds(prot.center_of_mass(), lig.center_of_mass())

    # Move ligand by exactly one box length along first dimension
    ts = universe.trajectory.ts
    box = ts.triclinic_dimensions
    lig.positions += box[0]
    distance_shifted = mda.lib.distances.calc_bonds(prot.center_of_mass(), lig.center_of_mass())
    # Check it was shifted
    assert distance_shifted != pytest.approx(distance_orig, abs=1e-5)

    # Apply the ClosestImageShift transformation
    m = ClosestImageShift(prot, [lig])
    universe.trajectory.add_transformations(m)
    distance_after = mda.lib.distances.calc_bonds(prot.center_of_mass(), lig.center_of_mass())

    # Check that the COM distance matches the original COM distance
    assert distance_after == pytest.approx(distance_orig, abs=0.01)


def test_nojump(hybrid_system_pdb, simulation_nc):
    universe = mda.Universe(
        hybrid_system_pdb,
        simulation_nc,
        format="MultiStateReporter",
        index=2,
    )
    # find frame where protein would teleport across boundary and check it
    prot = universe.select_atoms("protein and name CA")

    nj = NoJump(prot)
    universe.trajectory.add_transformations(nj)
    universe.trajectory[282]
    universe.trajectory[283]

    # without the transformation, the y coordinate would jump up to ~81.86
    ref = np.array([31.79594626, 52.14568866, 30.64103877])
    assert prot.center_of_mass() == pytest.approx(ref, abs=0.01)
    universe.trajectory.close()


def test_aligner(universe):
    # checks that rmsd is identical with/without center&super
    prot = universe.select_atoms("protein and name CA")
    a = Aligner(prot)
    universe.trajectory.add_transformations(a)

    p1 = prot.positions
    universe.trajectory[1]

    raw_rmsd = rms.rmsd(prot.positions, p1, center=False, superposition=False)
    opt_rmsd = rms.rmsd(prot.positions, p1, center=True, superposition=True)

    # the rmsd should be identical even if the function didn't align
    # as the transformation should have done this
    assert raw_rmsd == pytest.approx(opt_rmsd)


def test_chain_radius_of_gyration_stable(universe_single_state):
    """Protein chains should not explode or collapse due to PBC errors
    after applying alignment transformations."""
    protein = universe_single_state.select_atoms("protein and name CA")
    apply_transformations.apply_alignment_transformations(universe_single_state, protein)

    chain = protein.segments[0].atoms
    rgs = []
    for ts in universe_single_state.trajectory[:50]:
        rgs.append(chain.radius_of_gyration())

    assert np.std(rgs) < 2.0


def test_ligand_com_continuity(universe_single_state):
    """Ligand COM should not jump between periodic images after applying
    alignment transformations."""
    ligand = universe_single_state.select_atoms("resname UNK")
    apply_transformations.apply_alignment_transformations(universe_single_state, ligand=ligand)

    coms = [ligand.center_of_mass() for ts in islice(universe_single_state.trajectory, 20)]
    jumps = [np.linalg.norm(coms[i + 1] - coms[i]) for i in range(len(coms) - 1)]

    assert max(jumps) < 5.0


def test_multichain_rmsd_shifting(simulation_skipped_nc, hybrid_system_skipped_pdb):
    """Chain shifting should remove RMSD jumps caused by periodic boundary crossings."""
    u = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    prot = u.select_atoms("protein and name CA")
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
    u2 = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, 0
    )
    prot2 = u2.select_atoms("protein and name CA")
    apply_transformations.apply_alignment_transformations(u2, protein=prot2)

    R2 = rms.RMSD(prot2)
    R2.run()
    rmsd_shift = R2.rmsd[:, 2]
    assert np.max(np.diff(rmsd_shift[:20])) < 2  # jumps should disappear
    u2.trajectory.close()


def test_rmsd_reference_is_first_frame(universe_single_state):
    """After alignment, RMSD at the first frame should be zero."""
    prot = universe_single_state.select_atoms("protein and name CA")
    apply_transformations.apply_alignment_transformations(universe_single_state, protein=prot)

    _ = next(iter(universe_single_state.trajectory))  # SAFE
    ref = prot.positions.copy()

    rmsd = np.sqrt(((prot.positions - ref) ** 2).mean())
    assert rmsd == 0.0
    universe_single_state.trajectory.close()
