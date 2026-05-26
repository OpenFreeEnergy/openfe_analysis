import pytest

from openfe_analysis.rmsd import make_Universe
from openfe_analysis.utils.universe_utils import (
    correct_elements,
    guess_ligand_bonds,
    select_state_atoms,
)


@pytest.fixture
def universe(hybrid_system_skipped_pdb, simulation_skipped_nc):
    u = make_Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, state=0)
    yield u
    u.trajectory.close()


@pytest.fixture
def ligand_ag(universe):
    return select_state_atoms(universe, end_state="A").select_atoms("resname UNK")


def test_guess_ligand_bonds_adds_bonds(ligand_ag):
    """Bonds should be present on the atomgroup after guess_ligand_bonds."""
    original_count = len(ligand_ag.bonds)
    # This also has stateB bond
    assert original_count == 49
    guess_ligand_bonds(ligand_ag, delete_existing=True)
    # Now only 48 stateA bonds
    assert len(ligand_ag.bonds) == 48


def test_guess_ligand_bonds_modifies_universe_inplace(ligand_ag):
    """Bond topology should be reflected on the parent universe after guessing."""
    guess_ligand_bonds(ligand_ag)
    universe_bonds = ligand_ag.universe.select_atoms("resname UNK").bonds
    assert len(universe_bonds) > 0


@pytest.mark.parametrize(
    "end_state, expected_bfactors",
    [
        ("A", (0.25, 0.5)),
        ("B", (0.75, 0.5)),
    ],
)
def test_select_state_atoms(universe, end_state, expected_bfactors):
    """State selection should include state-unique and shared atoms."""
    state = select_state_atoms(universe, end_state=end_state)
    assert len(state) > 0
    assert all(atom.bfactor in expected_bfactors for atom in state)


def test_select_state_atoms_invalid_state(universe):
    """Invalid end_state should raise a ValueError."""
    with pytest.raises(ValueError, match="end_state must be 'A' or 'B'"):
        select_state_atoms(universe, end_state="C")


def test_select_state_atoms_shared_atoms(universe):
    """Shared atoms (bfactor 0.5) should appear in both state A and B selections."""
    state_a = select_state_atoms(universe, end_state="A")
    state_b = select_state_atoms(universe, end_state="B")
    shared_a = set(atom.ix for atom in state_a if atom.bfactor == 0.5)
    shared_b = set(atom.ix for atom in state_b if atom.bfactor == 0.5)
    assert shared_a == shared_b


def test_correct_elements_fixes_element():
    """correct_elements should update element where rdmol differs."""

    # Build a minimal universe with a C atom
    u = mda.Universe.empty(2, n_residues=1, trajectory=True)
    u.add_TopologyAttr("elements", ["C", "C"])  # second atom is wrong
    u.add_TopologyAttr("names", ["C1", "C2"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(
        np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]),
        order="fac",
    )
    ag = u.select_atoms("all")

    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))  # C
    mol.AddAtom(Chem.Atom(7))  # N
    rdmol = mol.GetMol()

    with pytest.warns(UserWarning, match="No atom_mapping provided"):
        correct_elements(ag, rdmol)

    assert ag[0].element == "C"
    assert ag[1].element == "N"
    assert ag[1].name == "N"


def test_correct_elements_no_change_when_correct():
    """correct_elements should not modify atoms that already have correct elements."""

    u = mda.Universe.empty(2, n_residues=1, trajectory=True)
    u.add_TopologyAttr("elements", ["C", "N"])
    u.add_TopologyAttr("names", ["C1", "N1"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(
        np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]),
        order="fac",
    )
    ag = u.select_atoms("all")

    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))  # C
    mol.AddAtom(Chem.Atom(7))  # N
    rdmol = mol.GetMol()

    with pytest.warns(UserWarning, match="No atom_mapping provided"):
        correct_elements(ag, rdmol)

    assert ag[0].element == "C"
    assert ag[0].name == "C1"  # name unchanged
    assert ag[1].element == "N"
    assert ag[1].name == "N1"  # name unchanged


def test_correct_elements_with_atom_mapping():
    """correct_elements with atom_mapping should use mapping"""

    u = mda.Universe.empty(2, n_residues=1, trajectory=True)
    u.add_TopologyAttr("elements", ["C", "C"])  # second atom is wrong
    u.add_TopologyAttr("names", ["C1", "C2"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(
        np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]),
        order="fac",
    )
    ag = u.select_atoms("all")

    # rdmol has atoms in reverse order: N, C
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(7))  # N at rdmol index 0
    mol.AddAtom(Chem.Atom(6))  # C at rdmol index 1
    rdmol = mol.GetMol()

    # explicitly map ag index 0 -> rdmol index 1 (C), ag index 1 -> rdmol index 0 (N)
    correct_elements(ag, rdmol, atom_mapping={0: 1, 1: 0})

    assert ag[0].element == "C"  # mapped to rdmol index 1 (C)
    assert ag[1].element == "N"  # mapped to rdmol index 0 (N)
    assert ag[1].name == "N"


def test_correct_elements_raises_size_error():
    """correct_elements should raise ValueError if atom counts don't match."""

    u = mda.Universe.empty(2, n_residues=1, trajectory=True)
    u.add_TopologyAttr("elements", ["C", "N"])
    u.add_TopologyAttr("names", ["C1", "N1"])
    u.add_TopologyAttr("resnames", ["UNK"])
    u.add_TopologyAttr("resids", [1])
    u.load_new(np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]), order="fac")
    ag = u.select_atoms("all")

    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))  # only 1 atom
    rdmol = mol.GetMol()

    with pytest.raises(ValueError, match="atomgroup has 2 atoms but rdmol has 1"):
        correct_elements(ag, rdmol)
