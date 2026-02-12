"""Transformations

Many on-the-fly transformations which are used to manipulate trajectories as
they are read.  This allows a trajectory to avoid periodic-boundary issues
and to automatically align the system to a protein structure.
"""

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.transformations.base import TransformationBase
from numpy import typing as npt


class NoJump(TransformationBase):
    """
    Prevent an AtomGroup from jumping between periodic images.

    This on-the-fly trajectory transformation removes large apparent
    center-of-mass displacements caused by periodic boundary conditions.
    If the AtomGroup moves by more than half a box length between
    consecutive frames, it is translated by an integer number of box
    vectors to keep its motion continuous.

    The transformation operates in-place on the AtomGroup coordinates
    and is intended to be applied before analyses that rely on smooth
    time evolution (e.g. RMSD, COM motion).

    Parameters
    ----------
    ag : MDAnalysis.AtomGroup
        AtomGroup whose center-of-mass motion should be made continuous.

    Notes
    -----
    - This transformation assumes an orthorhombic unit cell.
    - Only translations are applied; no rotations or scaling.
    - The correction is based on center-of-mass motion and is therefore
      most appropriate for compact groups (e.g. proteins, ligands).
    - Must be applied before any alignment transformations to avoid
      mixing reference frames.
    """

    ag: mda.AtomGroup
    prev: npt.NDArray

    def __init__(self, ag: mda.AtomGroup):
        super().__init__()
        self.ag = ag
        self.prev = ag.center_of_mass()

    def _transform(self, ts):
        if ts.frame == 0:
            self.prev = self.ag.center_of_mass()
        else:
            box = self.ag.dimensions[:3]
            current_position = self.ag.center_of_mass()

            diff = current_position - self.prev
            adjustment = box * np.rint(diff / box)

            self.ag.positions -= adjustment
            self.prev = self.ag.center_of_mass()

        return ts


class Minimiser(TransformationBase):
    """
    Translate AtomGroups to the nearest periodic image relative to a reference.

    This transformation shifts one or more AtomGroups by integer multiples
    of the simulation box vectors such that their center of mass is as close
    as possible to the center of mass of a reference AtomGroup.

    It is commonly used to keep ligands in the same periodic image as a
    protein during alchemical or replica-exchange simulations.

    Parameters
    ----------
    central_ag : MDAnalysis.AtomGroup
        Reference AtomGroup whose center of mass defines the target image.
    *ags : MDAnalysis.AtomGroup
        One or more AtomGroups to be translated into the closest periodic
        image relative to ``central_ag``.

    Notes
    -----
    - This transformation assumes an orthorhombic simulation box.
    - Translations are applied independently for each AtomGroup.
    - Coordinates are modified in-place.
    - This transformation does not prevent inter-frame jumps by itself
      and is typically used in combination with :class:`NoJump`.
    """

    central_ag: mda.AtomGroup
    other_ags: list[mda.AtomGroup]

    def __init__(self, central_ag: mda.AtomGroup, *ags):
        super().__init__()
        self.central_ag = central_ag
        self.other_ags = ags

    def _transform(self, ts):
        center = self.central_ag.center_of_mass()
        box = self.central_ag.dimensions[:3]

        for ag in self.other_ags:
            vec = ag.center_of_mass() - center

            # this only works for orthogonal boxes
            ag.positions -= np.rint(vec / box) * box

        return ts


class Aligner(TransformationBase):
    """
    Align a trajectory to a reference AtomGroup by minimizing RMSD.

    This transformation performs an on-the-fly least-squares alignment
    of the entire universe to a reference AtomGroup.
    At each frame, the coordinates are translated and rotated to minimize the
    RMSD of the atoms relative to their positions in the reference.
    """

    ref_pos: npt.NDArray
    ref_idx: npt.NDArray
    weights: npt.NDArray

    def __init__(self, ref_ag: mda.AtomGroup):
        super().__init__()
        self.ref_idx = ref_ag.ix
        self.ref_pos = ref_ag.positions
        self.weights = np.asarray(ref_ag.masses, dtype=np.float64)
        self.weights /= np.mean(self.weights)  # normalise weights
        # remove COM shift from reference positions
        self.ref_pos -= np.average(self.ref_pos, axis=0, weights=self.weights)

    def _transform(self, ts):
        # todo: worry about first frame?  can skip if ts.frame == 0?
        mobile_pos = ts.positions[self.ref_idx]
        mobile_com = np.average(mobile_pos, axis=0, weights=self.weights)

        mobile_pos -= mobile_com

        # rotates mobile to best align with ref
        R, min_rmsd = rotation_matrix(mobile_pos, self.ref_pos, weights=self.weights)

        # apply the transformation onto **all** atoms
        ts.positions -= mobile_com
        ts.positions = np.dot(ts.positions, R.T)

        return ts
