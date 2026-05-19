import MDAnalysis as mda
import pytest
from MDAnalysisTests.datafiles import DCD, PSF
from numpy.testing import assert_allclose, assert_almost_equal

from openfe_analysis.rmsd import RMSDAnalysis


@pytest.fixture
def mda_universe():
    return mda.Universe(PSF, DCD)


@pytest.fixture()
def correct_values():
    return [0, 4.68953]


@pytest.fixture()
def correct_values_mass():
    return [0, 4.74920]


def test_rmsd(mda_universe, correct_values):
    prot = mda_universe.select_atoms("name CA")
    prot_rmsd = RMSDAnalysis(prot, superposition=True).run(step=49)
    assert_almost_equal(
        prot_rmsd.results.rmsd,
        correct_values,
        4,
        err_msg="error: rmsd profile should match" + "test values",
    )


def test_rmsd_frames(mda_universe, correct_values):
    prot = mda_universe.select_atoms("name CA")
    prot_rmsd = RMSDAnalysis(prot, superposition=True).run(frames=[0, 49])
    assert_almost_equal(
        prot_rmsd.results.rmsd,
        correct_values,
        4,
        err_msg="error: rmsd profile should match" + "test values",
    )


def test_rmsd_single_frame(mda_universe):
    prot = mda_universe.select_atoms("name CA")
    prot_rmsd = RMSDAnalysis(prot, superposition=True).run(start=5, stop=6)
    single_frame = [0.91544906]
    assert_almost_equal(
        prot_rmsd.results.rmsd,
        single_frame,
        4,
        err_msg="error: rmsd profile should match" + "test values",
    )


def test_mass_weighted(mda_universe, correct_values):
    # mass weighting the CA should give the same answer as weighing
    # equally because all CA have the same mass
    prot = mda_universe.select_atoms("name CA")
    prot_rmsd = RMSDAnalysis(prot, superposition=True, mass_weighted=True).run(step=49)

    assert_almost_equal(
        prot_rmsd.results.rmsd,
        correct_values,
        4,
        err_msg="error: rmsd profile should matchtest values",
    )


def test_custom_weighted(mda_universe, correct_values_mass):
    prot = mda_universe.select_atoms("all")
    prot_rmsd = RMSDAnalysis(prot, superposition=True, mass_weighted=True).run(step=49)
    assert_almost_equal(
        prot_rmsd.results.rmsd,
        correct_values_mass,
        4,
        err_msg="error: rmsd profile should matchtest values",
    )
