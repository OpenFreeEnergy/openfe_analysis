from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import MDAnalysis as mda
import netCDF4 as nc
from MDAnalysis.guesser.tables import vdwradii as MDA_VDWRADII
from rdkit import Chem

from ..reader import FEReader

# B-factor values used to identify atoms present at a given lambda state.
# 0.25 : atoms unique to state A
# 0.75 : atoms unique to state B
# 0.5 : atoms shared by both end states.
_BFACTOR_STATE_A = (0.25, 0.5)
_BFACTOR_STATE_B = (0.75, 0.5)


def select_state_atoms(
    universe: mda.Universe,
    end_state: Literal["A", "B"],
) -> mda.AtomGroup:
    """
    Select all atoms present at a given end state.

    Atoms are identified by their b-factor values:

    - ``0.25`` — unique to state A
    - ``0.75`` — unique to state B
    - ``0.5``  — shared by both end states

    Parameters
    ----------
    universe : mda.Universe
        Universe containing the hybrid topology.
    end_state : {"A", "B"}
        The end state to select atoms for.

    Returns
    -------
    mda.AtomGroup
        All atoms present at the given end state.

    Raises
    ------
    ValueError
        If ``end_state`` is not ``"A"`` or ``"B"``.
    """
    if end_state == "A":
        bfactor_values = _BFACTOR_STATE_A
    elif end_state == "B":
        bfactor_values = _BFACTOR_STATE_B
    else:
        raise ValueError(f"end_state must be 'A' or 'B', got '{end_state}'")

    state = sum(
        [universe.atoms[universe.atoms.tempfactors == i] for i in bfactor_values],
        universe.atoms[[]],  # empty AtomGroup as start value
    )
    return state


def guess_ligand_bonds(
    atomgroup: mda.AtomGroup,
    delete_existing: bool = False,
) -> None:
    """
    Guess bonds for a ligand AtomGroup in-place.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Ligand atoms for which bonds will be guessed.
    delete_existing : bool, optional
        If ``True``, delete existing bonds on the atomgroup before guessing.
        This may be necessary to avoid cross-state bonds in hybrid topologies.
        Default is ``False``.
    """
    if delete_existing:
        atomgroup.universe.delete_bonds(atomgroup.bonds)
    # MDA vdw radii use uppercase element symbols (e.g. "CL", "BR", "NA"),
    # but RDKit uses mixed case; add aliases so bond guessing works correctly
    vdwradii = dict(MDA_VDWRADII)
    vdwradii.update(
        {
            "Cl": vdwradii["CL"],
            "Br": vdwradii["BR"],
            "Na": vdwradii["NA"],
        }
    )
    atomgroup.guess_bonds(vdwradii)



def select_state_atoms(
    universe: mda.Universe,
    end_state: Literal["A", "B"],
) -> mda.AtomGroup:
    """
    Select all atoms present at a given end state.

    Atoms are identified by their b-factor values:

    - ``0.25`` — unique to state A
    - ``0.75`` — unique to state B
    - ``0.5``  — shared by both end states

    Parameters
    ----------
    universe : mda.Universe
        Universe containing the hybrid topology.
    end_state : {"A", "B"}
        The end state to select atoms for.

    Returns
    -------
    mda.AtomGroup
        All atoms present at the given end state.

    Raises
    ------
    ValueError
        If ``end_state`` is not ``"A"`` or ``"B"``.
    """
    if end_state == "A":
        bfactor_values = _BFACTOR_STATE_A
    elif end_state == "B":
        bfactor_values = _BFACTOR_STATE_B
    else:
        raise ValueError(f"end_state must be 'A' or 'B', got '{end_state}'")

    state = sum(
        [universe.atoms[universe.atoms.tempfactors == i] for i in bfactor_values],
        universe.atoms[[]],  # empty AtomGroup as start value
    )
    return state


def guess_ligand_bonds(
    atomgroup: mda.AtomGroup,
    delete_existing: bool = False,
) -> None:
    """
    Guess bonds for a ligand AtomGroup in-place.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Ligand atoms for which bonds will be guessed.
    delete_existing : bool, optional
        If ``True``, delete existing bonds on the atomgroup before guessing.
        This may be necessary to avoid cross-state bonds in hybrid topologies.
        Default is ``False``.
    """
    if delete_existing:
        atomgroup.universe.delete_bonds(atomgroup.bonds)
    # MDA vdw radii use uppercase element symbols (e.g. "CL", "BR", "NA"),
    # but RDKit uses mixed case; add aliases so bond guessing works correctly
    vdwradii = dict(MDA_VDWRADII)
    vdwradii.update(
        {
            "Cl": vdwradii["CL"],
            "Br": vdwradii["BR"],
            "Na": vdwradii["NA"],
        }
    )
    atomgroup.guess_bonds(vdwradii)


def correct_elements(
    atomgroup: mda.AtomGroup,
    rdmol: Chem.Mol,
    atom_mapping: dict[int, int] | None = None,
) -> None:
    """
    Correct element and atom names in an AtomGroup in-place
    using an RDKit molecule.

    This is needed for hybrid topologies where mapped atoms that
    undergo element changes carry state A's element types, even when
    state B's ligand is selected.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Ligand atoms whose elements and names will be corrected.
    rdmol : Chem.Mol
        RDKit molecule with the correct element and atom name information.
    atom_mapping : dict[int, int], optional
        A mapping of ``{atomgroup_index: rdmol_index}`` defining the
        correspondence between atoms in ``atomgroup`` and ``rdmol``. If
        ``None``, atoms are matched by position which gives wrong results if
        the atom order was not the same.

    Raises
    ------
    ValueError
        If the number of atoms in ``atomgroup`` and ``rdmol`` do not match.
    """
    periodic_table = Chem.GetPeriodicTable()

    if len(atomgroup) != rdmol.GetNumAtoms():
        raise ValueError(
            f"atomgroup has {len(atomgroup)} atoms but rdmol has {rdmol.GetNumAtoms()} atoms."
        )

    if atom_mapping is not None:
        for ag_idx, rd_idx in atom_mapping.items():
            mda_atom = atomgroup[ag_idx]
            rd_atom = rdmol.GetAtomWithIdx(rd_idx)
            element = periodic_table.GetElementSymbol(rd_atom.GetAtomicNum())
            if mda_atom.element != element:
                mda_atom.element = element
                mda_atom.name = rd_atom.GetSymbol()
    else:
        warnings.warn(
            "No atom_mapping provided to correct_elements. Assuming that "
            "atom ordering is the same between atomgroup and rdmol. This may "
            "give incorrect results if the atom ordering differs between the two.",
            UserWarning,
        )
        for mda_atom, rd_atom in zip(atomgroup, rdmol.GetAtoms()):
            element = periodic_table.GetElementSymbol(rd_atom.GetAtomicNum())
            if mda_atom.element != element:
                mda_atom.element = element
                mda_atom.name = rd_atom.GetSymbol()
                
def create_universe_single_state(
    top: Path | mda.core.topology.Topology, trj: nc.Dataset, state: int
) -> mda.Universe:
    """
    Construct a raw MDAnalysis Universe for a single thermodynamic state.

    Parameters
    ----------
    top : pathlib.Path | mda.core.topology.Topology
        Path to a topology file (e.g. PDB).
    trj : nc.Dataset
        Open NetCDF dataset produced by
        ``openmmtools.multistate.MultiStateReporter``.
    state : int
        Thermodynamic state index to extract from the multistate trajectory.

    Returns
    -------
    mda.Universe
        A Universe with no trajectory transformations applied.
    """
    return mda.Universe(
        top,
        trj,
        index=state,
        index_method="state",
        format=FEReader,
    )
