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
        group = protein
        if has_ligand:
            group = protein + ligand
        # Unwrap all atoms
        unwrap_tr = unwrap(group)

        # Shift chains + ligand
        chains = [seg.atoms for seg in protein.segments]
        shift_targets = [*chains[1:]]
        if has_ligand:
            shift_targets.append(ligand)
        shift = ClosestImageShift(chains[0], shift_targets)

        align = Aligner(protein)

        u.trajectory.add_transformations(
            unwrap_tr,
            shift,
            align,
        )
    elif has_ligand:
        # if there's no protein
        # - make the ligand not jump periodic images between frames
        # - align the ligand to minimise its RMSD
        nope = NoJump(ligand)
        align = Aligner(ligand)

        u.trajectory.add_transformations(
            nope,
            align,
        )
