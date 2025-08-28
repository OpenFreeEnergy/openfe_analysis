import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import pytest

from openfe_analysis import FEReader
from openfe_analysis.transformations import (
    Minimiser,
    NoJump,
    Aligner,
)


@pytest.fixture
def universe(hybrid_system_pdb, simulation_nc):
    return mda.Universe(
        hybrid_system_pdb,
        simulation_nc,
        format="MultiStateReporter",
        state_id=0,
    )


@pytest.mark.flaky(reruns=3)
def test_minimiser(universe):
    prot = universe.select_atoms("protein and name CA")
    lig = universe.select_atoms("resname UNK")
    m = Minimiser(prot, lig)
    universe.trajectory.add_transformations(m)

    d = mda.lib.distances.calc_bonds(prot.center_of_mass(), lig.center_of_mass())
    # in the raw trajectory this is ~71 A as they're in diff images
    # accounting for pbc should result in ~11.10
    assert d == pytest.approx(11.10, abs=0.01)


@pytest.mark.flaky(reruns=3)
def test_nojump(universe):
    # find frame where protein would teleport across boundary and check it
    prot = universe.select_atoms("protein and name CA")

    nj = NoJump(prot)
    universe.trajectory.add_transformations(nj)

    universe.trajectory[169]
    universe.trajectory[170]

    # without the transformation, the y coordinate would jump up to ~81.86
    ref = np.array([72.37, -0.27, 66.49])
    assert prot.center_of_mass() == pytest.approx(ref, abs=0.01)


@pytest.mark.flaky(reruns=3)
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
