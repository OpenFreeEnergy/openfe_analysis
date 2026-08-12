import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase


class BoreschRestraintAnalysis(AnalysisBase):
    """
    Boresch restraint geometry (bond, 2 angles, 3 dihedrals) time series.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Exactly 6 atoms, ordered [H0, H1, H2, G0, G1, G2] (host atoms then
        guest atoms), matching the atom ordering used by
        ``openfe.protocols.restraint_utils.geometry.boresch``.

    Raises
    ------
    ValueError
        If ``atomgroup`` does not contain exactly 6 atoms.

    Notes
    -----
    Bond length is reported in Angstrom; angles and dihedrals are reported
    in radians, matching the units returned by MDAnalysis's
    ``calc_bonds``/``calc_angles``/``calc_dihedrals``. Reported quantities
    are, in order: the H0-G0 bond, the H1-H0-G0 and H0-G0-G1 angles, and the
    H2-H1-H0-G0, H1-H0-G0-G1, and H0-G0-G1-G2 dihedrals.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(self, atomgroup: mda.AtomGroup, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        if len(atomgroup) != 6:
            raise ValueError(
                f"atomgroup must contain exactly 6 atoms (3 host + 3 guest), got {len(atomgroup)}."
            )
        self._ag = atomgroup

    def _prepare(self) -> None:
        self.results.bond = np.zeros(self.n_frames, dtype=np.float64)
        self.results.angle1 = np.zeros(self.n_frames, dtype=np.float64)
        self.results.angle2 = np.zeros(self.n_frames, dtype=np.float64)
        self.results.dihedral1 = np.zeros(self.n_frames, dtype=np.float64)
        self.results.dihedral2 = np.zeros(self.n_frames, dtype=np.float64)
        self.results.dihedral3 = np.zeros(self.n_frames, dtype=np.float64)

    def _single_frame(self) -> None:
        atoms = self._ag.atoms
        box = self._ag.dimensions

        self.results.bond[self._frame_index] = mda.lib.distances.calc_bonds(
            atoms[0].position,
            atoms[3].position,
            box=box,
        )

        self.results.angle1[self._frame_index] = mda.lib.distances.calc_angles(
            atoms[1].position,
            atoms[0].position,
            atoms[3].position,
            box=box,
        )
        self.results.angle2[self._frame_index] = mda.lib.distances.calc_angles(
            atoms[0].position,
            atoms[3].position,
            atoms[4].position,
            box=box,
        )

        self.results.dihedral1[self._frame_index] = mda.lib.distances.calc_dihedrals(
            atoms[2].position,
            atoms[1].position,
            atoms[0].position,
            atoms[3].position,
            box=box,
        )
        self.results.dihedral2[self._frame_index] = mda.lib.distances.calc_dihedrals(
            atoms[1].position,
            atoms[0].position,
            atoms[3].position,
            atoms[4].position,
            box=box,
        )
        self.results.dihedral3[self._frame_index] = mda.lib.distances.calc_dihedrals(
            atoms[0].position,
            atoms[3].position,
            atoms[4].position,
            atoms[5].position,
            box=box,
        )
