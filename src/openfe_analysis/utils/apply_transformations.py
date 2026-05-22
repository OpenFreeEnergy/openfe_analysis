from __future__ import annotations

import MDAnalysis as mda
from MDAnalysis.transformations import unwrap

from ..transformations import Aligner, ClosestImageShift, NoJump


def apply_alignment_transformations(
    universe: mda.Universe,
    protein: mda.AtomGroup | None = None,
    ligand: mda.AtomGroup | None = None,
) -> None:
    """
    Apply a standard set of PBC-handling and alignment transformations
    required for RMSD-based analyses and other structural analyses that
    assume a pre-processed trajectory.

    Parameters
    ----------
    universe: mda.Universe
        The Universe the transformations are applied to. Modified in-place.
    protein: mda.AtomGroup | None
        The AtomGroup of the protein
    ligand: mda.AtomGroup | None
        The AtomGroup of the ligand

    Notes
    -----
    Depending on whether a protein is present, a sequence of trajectory
    transformations is applied:

    If a protein is present:

    - Unwraps protein and ligand atom to be made whole
    - Shifts protein chains and the ligand to the image closest to the first
      protein chain (:class:`ClosestImageShift`)
    - Aligns the entire system to minimize the protein RMSD (:class:`Aligner`)

    If only a ligand is present:

    - Prevents the ligand from jumping between periodic images
    - Aligns the ligand to minimize its RMSD

    If neither protein nor ligand is provided, no transformations are applied.
    """
    has_protein = protein is not None and protein.n_atoms > 0
    has_ligand = ligand is not None and ligand.n_atoms > 0

    if has_protein:
        lig = ligand if has_ligand else None
        transforms = _transformations_complex(protein, lig)
    elif has_ligand:
        transforms = _transformations_ligand_only(ligand)
    else:
        return

    universe.trajectory.add_transformations(*transforms)


def _transformations_complex(
    protein: mda.AtomGroup,
    ligand: mda.AtomGroup | None = None,
) -> list:
    """
    Build transformations for systems containing a protein and optionally
    a ligand.

    Parameters
    ----------
    protein : mda.AtomGroup
        Protein atoms to use for alignment and image shifting.
    ligand :  mda.AtomGroup | None
        Ligand atoms. If provided, included in unwrapping and image shifting.

    Returns
    -------
    list
        Ordered list of trajectory transformations to apply.
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


def _transformations_ligand_only(ligand: mda.AtomGroup) -> list:
    """
    Build transformations for ligand-only systems.

    Parameters
    ----------
    ligand : mda.AtomGroup
        Ligand atoms to apply transformations to.

    Returns
    -------
    list
        Ordered list of trajectory transformations to apply:

        - Prevent the ligand from jumping between periodic images
        - Align the ligand to minimize its RMSD
    """
    return [
        NoJump(ligand),
        Aligner(ligand),
    ]
