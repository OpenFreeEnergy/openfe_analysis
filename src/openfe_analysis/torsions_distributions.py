import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import netCDF4 as nc
import numpy as np
import pathlib

from openfe_analysis.reader import FEReader
from openfe_analysis.transformations import (
    NoJump, Minimiser, Aligner
)

from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem

from matplotlib import pyplot as plt
from MDAnalysis.analysis.dihedrals import Dihedral

def calculate_dihedrals(mol, torsion_ids_list)->np.array:
    dihedrals = []
    for c in mol.GetConformers():
        frame_dihs = []
        for tba in torsion_ids_list:
            tangle = Chem.rdMolTransforms.GetDihedralDeg(c, *tba)
            frame_dihs.append(tangle)
        dihedrals.append(frame_dihs)
    dihedrals = np.array(dihedrals_B)
    return dihedrals


def calculate_dihedrals_all_torsions(mol)->(List[List[int]], np.array):
    rbA = get_rotatable_bonds(mol)

    tbA = []
    for rba in rbA:
        tba = get_torsion_atoms_idx(rba)
        tbA.append(tba)

    dihedrals = calculate_dihedrals(mol, tbA)

    return (tbA, dihedrals)


def get_rotatable_bonds(mol: Chem.Mol) -> List[Chem.Bond]:
    """Function to find all rotatable bonds in a molecule taking symmetry into account.

    Parameters
    ----------
    mol : Chem.Mol
        The rdkit.Chem.Mol object

    Returns
    -------
    List[Chem.Bond]
         List of rdkit.Chem.Bond that were found in a molecule taking symmetry into account.
    """
    RotatableBondSmarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    find_rbs = lambda x, y=RotatableBondSmarts: x.GetSubstructMatches(y, uniquify=1)
    rbs = find_rbs(mol)
    bonds = [mol.GetBondBetweenAtoms(*inds) for inds in rbs]

    return bonds


def get_torsion_atoms_idx(bond: Chem.Bond) -> List[int]:
    """Function that finds the atomic ids that specify a torsion around the bond of interest.

    Parameters
    ----------
    bond : Chem.Bond
        The bond of interest around which the torsion should be specified.

    Returns
    -------
    List[int]
        List of atomic ids that specify a torsion around the bond of interest.
    """
    bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
    additional_atom1 = list(filter(lambda x: x.GetIdx() != bond_atoms[1].GetIdx(), bond_atoms[0].GetNeighbors()))[0]
    additional_atom2 = list(filter(lambda x: x.GetIdx() != bond_atoms[0].GetIdx(), bond_atoms[1].GetNeighbors()))[0]
    torsion_atoms = [additional_atom1] + list(bond_atoms) + [additional_atom2]
    torsion_atom_ids = [a.GetIdx() for a in torsion_atoms]

    return torsion_atom_ids

