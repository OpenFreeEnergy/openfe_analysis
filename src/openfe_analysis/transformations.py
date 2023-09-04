import numpy as np
from MDAnalysis.transformations.base import TransformationBase
from MDAnalysis.analysis.align import rotation_matrix


class NoJump(TransformationBase):
    """Stops an AtomGroup from moving more than half a box length between frames"""
    def __init__(self, ag):
        super().__init__()
        self.ag = ag
        self.prev = ag.center_of_mass()

    def _transform(self, ts):
        if ts.frame != 0:
            box = self.ag.dimensions[:3]
            current_position = self.ag.center_of_mass()

            diff = current_position - self.prev
            adjustment = box * np.rint(diff / box)

            self.ag.positions -= adjustment
            self.prev = self.ag.center_of_mass()

        return ts


class Minimiser(TransformationBase):
    """Minimises the difference from ags to central_ag by choosing image"""
    def __init__(self, central_ag, *ags):
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
    """On-the-fly transformation to align a trajectory to minimise RMSD

    centers all coordinates onto origin
    rotates entire universe to minimise rmsd of **ref_ag**
    """
    ref_pos: np.ndarray
    ref_idx: np.ndarray
    weights: np.ndarray

    def __init__(self, ref_ag):
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
        R, min_rmsd = rotation_matrix(mobile_pos, self.ref_pos,
                                      weights=self.weights)

        # apply the transformation onto **all** atoms
        ts.positions -= mobile_com
        ts.positions = np.dot(ts.positions, R.T)

        return ts
