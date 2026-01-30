import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis import rms

from openfe_analysis import FEReader
from openfe_analysis.transformations import (
    Aligner,
    ClosestImageShift,
    NoJump,
)


@pytest.fixture
def universe(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = mda.Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        format="MultiStateReporter",
        state_id=0,
    )
    yield u
    u.trajectory.close()


def test_closest_image_shift(universe):
    prot = universe.select_atoms("protein and name CA")
    lig = universe.select_atoms("resname UNK")
    m = ClosestImageShift(prot, [lig])
    universe.trajectory.add_transformations(m)

    d = mda.lib.distances.calc_bonds(prot.center_of_mass(), lig.center_of_mass())
    # in the raw trajectory this is ~71 A as they're in diff images
    # accounting for pbc should result in ~11.10
    # TODO: This will be updated in the next PR!!!!
    assert d == pytest.approx(24.79, abs=0.01)


def test_nojump(hybrid_system_pdb, simulation_nc):
    universe = mda.Universe(
        hybrid_system_pdb,
        simulation_nc,
        format="MultiStateReporter",
        state_id=2,
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
