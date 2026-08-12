import MDAnalysis as mda
import pytest
from MDAnalysis.lib.distances import calc_angles, calc_bonds, calc_dihedrals
from MDAnalysisTests.datafiles import DCD, PSF
from numpy.testing import assert_allclose

from openfe_analysis.restraints import BoreschRestraintAnalysis


@pytest.fixture
def mda_universe():
    return mda.Universe(PSF, DCD)


class TestBoreschRestraintAnalysis:
    def test_matches_direct_mda_calls(self, mda_universe):
        atoms = mda_universe.atoms[[10, 20, 30, 40, 50, 60]]
        result = BoreschRestraintAnalysis(atoms).run(step=25)

        expected = {
            "bond": [],
            "angle1": [],
            "angle2": [],
            "dihedral1": [],
            "dihedral2": [],
            "dihedral3": [],
        }
        for _ in mda_universe.trajectory[::25]:
            box = atoms.dimensions
            expected["bond"].append(calc_bonds(atoms[0].position, atoms[3].position, box=box))
            expected["angle1"].append(
                calc_angles(atoms[1].position, atoms[0].position, atoms[3].position, box=box)
            )
            expected["angle2"].append(
                calc_angles(atoms[0].position, atoms[3].position, atoms[4].position, box=box)
            )
            expected["dihedral1"].append(
                calc_dihedrals(
                    atoms[2].position,
                    atoms[1].position,
                    atoms[0].position,
                    atoms[3].position,
                    box=box,
                )
            )
            expected["dihedral2"].append(
                calc_dihedrals(
                    atoms[1].position,
                    atoms[0].position,
                    atoms[3].position,
                    atoms[4].position,
                    box=box,
                )
            )
            expected["dihedral3"].append(
                calc_dihedrals(
                    atoms[0].position,
                    atoms[3].position,
                    atoms[4].position,
                    atoms[5].position,
                    box=box,
                )
            )

        assert_allclose(result.results.bond, expected["bond"])
        assert_allclose(result.results.angle1, expected["angle1"])
        assert_allclose(result.results.angle2, expected["angle2"])
        assert_allclose(result.results.dihedral1, expected["dihedral1"])
        assert_allclose(result.results.dihedral2, expected["dihedral2"])
        assert_allclose(result.results.dihedral3, expected["dihedral3"])

    @pytest.mark.parametrize("n_atoms", [5, 7])
    def test_raises_on_wrong_atom_count(self, mda_universe, n_atoms):
        with pytest.raises(ValueError, match="exactly 6 atoms"):
            BoreschRestraintAnalysis(mda_universe.atoms[:n_atoms])
