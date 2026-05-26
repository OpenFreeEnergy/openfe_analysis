from functools import partial

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis import diffusionmap, rms
from MDAnalysisTests.datafiles import DCD, PSF
from numpy.testing import assert_allclose, assert_almost_equal

from openfe_analysis.rmsd import (
    LigandCOMDrift,
    Protein2DRMSD,
    RMSDAnalysis,
    SymmetryCorrectedLigandRMSD,
    gather_rms_data,
    make_Universe,
)
from openfe_analysis.utils import universe_utils


@pytest.fixture
def mda_universe():
    return mda.Universe(PSF, DCD)


@pytest.fixture
def ligand(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = make_Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, state=0)
    yield u.select_atoms("resname UNK")
    u.trajectory.close()


@pytest.fixture()
def correct_values():
    return [0, 4.68953]


@pytest.fixture()
def correct_values_mass():
    return [0, 4.74920]


class TestRMSDAnalysis:
    def test_rmsd(self, mda_universe, correct_values):
        prot = mda_universe.select_atoms("name CA")
        prot_rmsd = RMSDAnalysis(prot, superposition=True).run(step=49)
        assert_almost_equal(
            prot_rmsd.results.rmsd,
            correct_values,
            4,
            err_msg="error: rmsd profile should match test values",
        )

    def test_rmsd_frames(self, mda_universe, correct_values):
        prot = mda_universe.select_atoms("name CA")
        prot_rmsd = RMSDAnalysis(prot, superposition=True).run(frames=[0, 49])
        assert_almost_equal(
            prot_rmsd.results.rmsd,
            correct_values,
            4,
            err_msg="error: rmsd profile should match test values",
        )

    def test_rmsd_single_frame(self, mda_universe):
        prot = mda_universe.select_atoms("name CA")
        prot_rmsd = RMSDAnalysis(prot, superposition=True).run(start=5, stop=6)
        assert_almost_equal(
            prot_rmsd.results.rmsd,
            [0.91544906],
            4,
            err_msg="error: rmsd profile should match test values",
        )

    def test_mass_weighted(self, mda_universe, correct_values):
        # mass weighting the CA should give the same answer as weighing
        # equally because all CA have the same mass
        prot = mda_universe.select_atoms("name CA")
        prot_rmsd = RMSDAnalysis(prot, superposition=True, mass_weighted=True).run(step=49)
        assert_almost_equal(
            prot_rmsd.results.rmsd,
            correct_values,
            4,
            err_msg="error: rmsd profile should match test values",
        )

    def test_custom_weighted(self, mda_universe, correct_values_mass):
        prot = mda_universe.select_atoms("all")
        prot_rmsd = RMSDAnalysis(prot, superposition=True, mass_weighted=True).run(step=49)
        assert_almost_equal(
            prot_rmsd.results.rmsd,
            correct_values_mass,
            4,
            err_msg="error: rmsd profile should match test values",
        )


class TestProtein2DRMSD:
    def test_output_shape(self, mda_universe):
        """Output should have n*(n-1)//2 entries for n frames."""
        prot = mda_universe.select_atoms("name CA")
        result = Protein2DRMSD(prot).run(step=10)
        n_frames = len(mda_universe.trajectory[::10])
        expected_pairs = n_frames * (n_frames - 1) // 2
        assert len(result.results.rmsd2d) == expected_pairs

    def test_values_nonnegative(self, mda_universe):
        """All RMSD values should be non-negative."""
        prot = mda_universe.select_atoms("name CA")
        result = Protein2DRMSD(prot).run(step=10)
        assert np.all(result.results.rmsd2d >= 0)

    def test_symmetric_diagonal_zero(self, mda_universe):
        """When reconstructed into a full matrix, diagonal should be zero
        and matrix should be symmetric."""
        prot = mda_universe.select_atoms("name CA")
        result = Protein2DRMSD(prot).run(step=10)

        n_frames = len(mda_universe.trajectory[::10])
        mat = np.zeros((n_frames, n_frames))
        mat[np.triu_indices_from(mat, k=1)] = result.results.rmsd2d
        mat += mat.T

        assert_allclose(np.diag(mat), 0.0)
        assert_allclose(mat, mat.T)

    def test_matches_mda_distance_matrix(self, mda_universe):
        """Results should match MDAnalysis DistanceMatrix."""
        prot = mda_universe.select_atoms("name CA")
        result = Protein2DRMSD(prot).run(step=10)

        metric = partial(rms.rmsd, center=True, superposition=True)
        ref = diffusionmap.DistanceMatrix(prot, metric=metric)
        ref.run(step=10)
        dist_mat = ref.results.dist_matrix
        i, j = np.triu_indices_from(dist_mat, k=1)
        expected = dist_mat[i, j]

        assert_allclose(result.results.rmsd2d, expected, atol=1e-4)


class TestLigandCOMDrift:
    def test_first_frame_is_zero(self, ligand):
        """COM drift at the first frame should always be zero."""
        result = LigandCOMDrift(ligand).run(step=10)
        assert result.results.com_drift[0] == pytest.approx(0.0, abs=1e-5)

    def test_output_shape(self, ligand):
        """Output should have one entry per analyzed frame."""
        result = LigandCOMDrift(ligand).run(step=10)
        n_frames = len(ligand.universe.trajectory[::10])
        assert len(result.results.com_drift) == n_frames

    def test_values_nonnegative(self, ligand):
        """COM drift values should be non-negative distances."""
        result = LigandCOMDrift(ligand).run(step=10)
        assert np.all(result.results.com_drift >= 0)


class TestSymmetryCorrectedLigandRMSD:
    def test_regression(self, ligand):
        state_lig = universe_utils.select_state_atoms(ligand.universe, end_state="A").select_atoms(
            "resname UNK"
        )
        result = SymmetryCorrectedLigandRMSD(state_lig).run(step=10)
        expected = [0.0, 0.75138, 2.09003, 0.95125, 1.54566, 2.00029]
        assert_allclose(result.results.rmsd[:6], expected, rtol=1e-3)

    def test_zero_for_valid_swap(self):
        """
        For a water-like symmetric molecule, swapping the two equivalent H atoms
        gives naive RMSD > 0 but SymmetryCorrectedLigandRMSD = 0.
        """
        # Build a minimal universe with two frames: reference and swapped
        coords_ref = np.array(
            [
                [0.0, 0.0, 0.0],  # O
                [1.0, 0.0, 0.0],  # H1
                [0.0, 1.0, 0.0],  # H2
            ]
        )
        coords_swapped = np.array(
            [
                [0.0, 0.0, 0.0],  # O
                [0.0, 1.0, 0.0],  # H2 in H1's slot
                [1.0, 0.0, 0.0],  # H1 in H2's slot
            ]
        )

        u = mda.Universe.empty(3, trajectory=True)
        u.add_TopologyAttr("elements", ["O", "H", "H"])
        u.add_TopologyAttr("names", ["O", "H1", "H2"])
        u.add_TopologyAttr("resnames", ["UNK"])
        u.add_TopologyAttr("resids", [1])
        u.load_new(
            np.array([coords_ref, coords_swapped]),
            order="fac",
        )

        ag = u.select_atoms("all")

        corrected = SymmetryCorrectedLigandRMSD(ag).run()
        naive = RMSDAnalysis(ag).run()

        # Frame 0 is reference — both should be 0
        assert corrected.results.rmsd[0] == pytest.approx(0.0, abs=1e-5)
        assert naive.results.rmsd[0] == pytest.approx(0.0, abs=1e-5)

        # Frame 1 is the swap — naive sees displacement, corrected sees zero
        assert naive.results.rmsd[1] > 0.0
        assert corrected.results.rmsd[1] == pytest.approx(0.0, abs=1e-5)
