from typing import Optional

import MDAnalysis as mda
from MDAnalysis.transformations import unwrap

from ..transformations import Aligner, ClosestImageShift, NoJump


def apply_transformations(
    u: mda.Universe,
    protein: Optional[mda.AtomGroup] = None,
    ligand: Optional[mda.AtomGroup] = None,
):
    """
    Apply a collection of transformations to a Universe.

    Parameters
    ----------
    u: Universe
        The Universe the transformations are applied to
    protein: Optional[AtomGroup]
        The AtomGroup of the protein
    ligand: Optional[AtomGroup]
        The AtomGroup of the ligand

    Notes
    -----
    Depending on whether a protein is present, a sequence of trajectory
    transformations is applied:

    If a protein is present:
    - Unwraps protein and ligand atom to be made whole
    - Shifts protein chains and the ligand to the image closest to the first
      protein chain (:class:`ClosestImageShift`)
    - Aligns the entire system to minimise the protein RMSD (:class:`Aligner`)

    If only a ligand is present:
    - Prevents the ligand from jumping between periodic images
    - Aligns the ligand to minimize its RMSD
    """
    has_protein = protein is not None and protein.n_atoms > 0
    has_ligand = ligand is not None and ligand.n_atoms > 0

    if has_protein:
        lig = ligand if has_ligand else None
        transforms = _apply_transformations_complex(protein, lig)
    elif has_ligand:
        transforms = _apply_transformations_ligand_only(ligand)
    else:
        return

    u.trajectory.add_transformations(*transforms)


def _apply_transformations_complex(protein, ligand=None):
    """
    Build transformations for systems containing a protein
    and optionally a ligand.
    """
    transforms = []
    # 1. Make molecules whole (protein + optional ligand)
    group = protein if ligand is None else protein + ligand
    transforms.append(unwrap(group))

    # 2. Closest image shift for protein chains + ligand (if present)
    chains = [seg.atoms for seg in protein.segments]
    shift_targets = chains[1:]
    if ligand is not None:
        shift_targets.append(ligand)
    transforms.append(ClosestImageShift(chains[0], shift_targets))

    # 3. Align on protein backbone/atoms
    transforms.append(Aligner(protein))

    return transforms


def _apply_transformations_ligand_only(ligand):
    """
    Build transformations for ligand-only systems.
      - make the ligand not jump periodic images between frames
      - align the ligand to minimize its RMSD
    """
    return [
        NoJump(ligand),
        Aligner(ligand),
    ]
