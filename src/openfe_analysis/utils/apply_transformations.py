from __future__ import annotations

import MDAnalysis as mda
from MDAnalysis.transformations import unwrap

from ..transformations import Aligner, ClosestImageShift, NoJump


def apply_complex_alignment_transformations(
    universe: mda.Universe,
    protein: mda.AtomGroup,
    ligands: list[mda.AtomGroup] | None = None,
) -> None:
    """
    Apply a standard set of PBC-handling and alignment transformations
    required for RMSD-based analyses and other structural analyses that
    assume a pre-processed trajectory.

    Parameters
    ----------
    universe: mda.Universe
        The Universe the transformations are applied to. Modified in-place.
    protein: mda.AtomGroup
        The AtomGroup of the protein
    ligands: list[mda.AtomGroup] | None
        List of ligand AtomGroups. Each is unwrapped and shifted to the
        closest image of the protein independently. If None or empty,
        only the protein is used.

    Raises
    ------
    ValueError
        If ``protein`` is None or contains no atoms.

    Notes
    -----
    The following transformations are applied in order:

    - Unwraps protein and all ligands to be made whole
    - Shifts protein chains and each ligand independently to the image
      closest to the first protein chain (:class:`ClosestImageShift`)
    - Aligns the entire system to minimize the protein RMSD (:class:`Aligner`)
    """
    if protein is None or not protein:
        raise ValueError("protein AtomGroup is empty or None")

    if not protein.bonds:
        raise ValueError(
            "protein AtomGroup has no bonds which would lead to wrong alignment. "
            "Call guess_bonds() on the protein before applying these transformations."
        )

    if isinstance(ligands, mda.AtomGroup):
        raise TypeError(
            "ligands must be a list of AtomGroups, not a single AtomGroup. "
            "Use ligands=[ligand] to wrap a single ligand."
        )

    ligands = [lig for lig in (ligands or []) if lig]

    group = protein
    for lig in ligands:
        group = group + lig
    # 1. Make molecules whole (protein + optional ligand)
    transforms = [unwrap(group)]

    # 2. Closest image shift for protein fragments + ligand (if present)
    fragments = list(protein.fragments)
    # Pick the largest fragment as the reference
    ref_idx = max(range(len(fragments)), key=lambda i: fragments[i].n_atoms)
    reference = fragments[ref_idx]
    shift_targets = [f for i, f in enumerate(fragments) if i != ref_idx]
    shift_targets += ligands
    if shift_targets:
        transforms.append(ClosestImageShift(reference=reference, targets=shift_targets))

    # 3. Align on protein backbone/atoms
    transforms.append(Aligner(protein))
    universe.trajectory.add_transformations(*transforms)


def apply_ligand_alignment_transformations(
    universe: mda.Universe,
    ligand: mda.AtomGroup,
) -> None:
    """
    Apply PBC-handling and alignment transformations for ligand-only systems.

    Parameters
    ----------
    universe : mda.Universe
        The Universe the transformations are applied to. Modified in-place.
    ligand : mda.AtomGroup
        Ligand atoms to apply transformations to.

    Raises
    ------
    ValueError
        If ``ligand`` is None or contains no atoms.

    Notes
    -----
    The following transformations are applied in order:

    - Prevents the ligand from jumping between periodic images
      (:class:`NoJump`)
    - Aligns the ligand to minimize its RMSD (:class:`Aligner`)
    """
    if ligand is None or ligand.n_atoms == 0:
        raise ValueError("ligand AtomGroup is empty or None")

    universe.trajectory.add_transformations(
        NoJump(ligand),
        Aligner(ligand),
    )
