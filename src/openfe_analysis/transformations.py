"""Transformations

Many on-the-fly transformations which are used to manipulate trajectories as
they are read.  This allows a trajectory to avoid periodic-boundary issues
and to automatically align the system to a protein structure.
"""

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.lib import distances
from MDAnalysis.transformations.base import TransformationBase
from numpy import typing as npt


class NoJump(TransformationBase):
    """Stops an AtomGroup from moving more than half a box length between frames

    This transformation prevents an AtomGroup "teleporting" across the box
    border between two subsequent frames.  This then simplifies the calculation
    of motion over time.
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


class ClosestImageShift(TransformationBase):
    """
    PBC-safe transformation that shifts one or more target AtomGroups
    so that their COM is in the closest image relative to a reference AtomGroup.
    Works for any box type (triclinic or orthorhombic).

    CAVEAT:
    This Transformation requires the AtomGroups to be unwrapped!

    Inspired from:
    https://github.com/wolberlab/OpenMMDL/blob/main/openmmdl/openmmdl_simulation/scripts/post_md_conversions.py
    """

    def __init__(self, reference: mda.AtomGroup, targets: list[mda.AtomGroup]):
        super().__init__()
        self.reference = reference
        self.targets = targets

    def _transform(self, ts):
        center = self.reference.center_of_mass()

        for ag in self.targets:
            vec = ag.center_of_mass() - center
            vec_min = distances.minimize_vectors(vec.reshape(1, 3), ts.dimensions)[0]
            ag.translate(vec_min - vec)

        return ts


class Aligner(TransformationBase):
    """On-the-fly transformation to align a trajectory to minimise RMSD

    centers all coordinates onto origin
    rotates **entire universe** to minimise rmsd relative to **ref_ag**
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
