from functools import partial

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import pytest
from MDAnalysis.analysis import diffusionmap, rms
from MDAnalysisTests.datafiles import DCD, PSF
from numpy.testing import assert_allclose, assert_almost_equal
from rdkit import Chem

from openfe_analysis.rmsd import (
    LigandCOMDrift,
    Protein2DRMSD,
    RMSDAnalysis,
    SymmetryCorrectedLigandRMSD,
)
from openfe_analysis.utils import apply_transformations, universe_utils


@pytest.fixture
def mda_universe():
    return mda.Universe(PSF, DCD)


@pytest.fixture
def ligand(hybrid_system_skipped_pdb, simulation_skipped_nc):
    universe = universe_utils.create_universe_single_state(
        hybrid_system_skipped_pdb, simulation_skipped_nc, state=0
    )
    prot = universe.select_atoms("protein and name CA")
    ligand = universe.select_atoms("resname UNK")
    apply_transformations.apply_complex_alignment_transformations(universe, prot, [ligand])
    yield universe.select_atoms("resname UNK")
    universe.trajectory.close()


@pytest.fixture
def minimal_universe():
    """Minimal 3-atom water-like universe without bonds."""
    u = mda.Universe.empty(3, n_residues=1, trajectory=True)
    u.add_TopologyAttr("elements", ["O", "H", "H"])
    u.add_TopologyAttr("names", ["O", "H1", "H2"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        order="fac",
    )
    return u.select_atoms("all")


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

    @pytest.mark.parametrize(
        "state_idx,ligand_key",
        [
            (14, "ligand_A_indices"),
            (12, "ligand_B_indices"),
            (16, "ligand_B_indices"),
        ],
    )
    def test_separate_ligands_fixes_pbc_spike(
        self,
        septop_complex_data,
        state_idx,
        ligand_key,
    ):
        """
        Regression test for PBC imaging artifact when two ligands are passed
        as a combined AtomGroup (SepTop case).

        Known problematic states from transformation HIF2a rbfe_1_155:
        - State 14, ligand A: ~62 A spike at frame 45 (4500 ps)
        - State 12, ligand B: ~62 A spike at frame 45 (4500 ps)
        - State 16, ligand B: ~62 A spike at frame 39 (3900 ps)

        Passing ligands separately to apply_complex_alignment_transformations
        fixes this by imaging each ligand independently relative to the protein.
        """
        d = septop_complex_data

        with nc.Dataset(d["nc"]) as ds:
            if hasattr(ds, "PositionInterval"):
                n_frames = len(range(0, ds.dimensions["iteration"].size, ds.PositionInterval))
            else:
                n_frames = ds.dimensions["iteration"].size
            skip = max(n_frames // 500, 1)

            # Combined approach should produce a spike
            u_combined = universe_utils.create_universe_single_state(d["pdb"], ds, state=state_idx)
            prot = u_combined.select_atoms("protein and name CA")
            lig_A = u_combined.atoms[d["ligand_A_indices"]]
            lig_B = u_combined.atoms[d["ligand_B_indices"]]
            lig = u_combined.atoms[d[ligand_key]]

            apply_transformations.apply_complex_alignment_transformations(
                u_combined, protein=prot, ligands=[lig_A + lig_B]
            )

            rmsd_combined = RMSDAnalysis(lig).run(step=skip)
            assert np.max(rmsd_combined.results.rmsd) > 10.0

            # Separate approach should fix it
            u_separate = universe_utils.create_universe_single_state(d["pdb"], ds, state=state_idx)
            prot = u_separate.select_atoms("protein and name CA")
            lig_A = u_separate.atoms[d["ligand_A_indices"]]
            lig_B = u_separate.atoms[d["ligand_B_indices"]]
            lig = u_separate.atoms[d[ligand_key]]

            apply_transformations.apply_complex_alignment_transformations(
                u_separate, protein=prot, ligands=[lig_A, lig_B]
            )

            rmsd_separate = RMSDAnalysis(lig).run(step=skip)
            assert np.max(rmsd_separate.results.rmsd) < 10.0


class TestProtein2DRMSD:
    def test_symmetric_diagonal_zero(self, mda_universe):
        """Diagonal should be zero and matrix should be symmetric."""
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

        # Check the output shape
        n_frames = len(mda_universe.trajectory[::10])
        expected_pairs = n_frames * (n_frames - 1) // 2
        assert len(result.results.rmsd2d) == expected_pairs
        assert np.all(result.results.rmsd2d >= 0)

        # Distancematrix doesn't do centering and superposition by default
        metric = partial(rms.rmsd, center=True, superposition=True)
        ref = diffusionmap.DistanceMatrix(prot, metric=metric)
        ref.run(step=10)
        dist_mat = ref.results.dist_matrix
        i, j = np.triu_indices_from(dist_mat, k=1)
        expected = dist_mat[i, j]

        assert_allclose(result.results.rmsd2d, expected, atol=1e-4)


def test_ligand_com_drift(ligand):
    result = LigandCOMDrift(ligand).run(step=10)
    expected = [0.0, 0.38549, 0.61483, 0.54140, 1.26861, 0.92772]
    assert len(result.results.com_drift) == len(ligand.universe.trajectory[::10])
    assert_allclose(result.results.com_drift[:6], expected, rtol=1e-3)


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
        u.add_TopologyAttr("bonds", [(0, 1), (0, 2)])
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

    def test_raises_on_missing_bonds(self, minimal_universe):
        """Should raise ValueError if atomgroup has no bonds and no rdmol is provided."""
        with pytest.raises(ValueError, match="No bonds found"):
            SymmetryCorrectedLigandRMSD(minimal_universe)

    def test_raises_on_atom_count_mismatch(self, minimal_universe):
        """Should raise ValueError if atomgroup and rdmol have different atom counts."""
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(8))  # O
        mol.AddAtom(Chem.Atom(1))  # H
        with pytest.raises(ValueError, match="atomgroup has 3 atoms but rdmol has 2"):
            SymmetryCorrectedLigandRMSD(minimal_universe, rdmol=mol.GetMol())
