import itertools
import pathlib
from typing import Any, Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations import unwrap

from .reader import FEReader
from .transformations import Aligner, ClosestImageShift, NoJump


def make_Universe(top: pathlib.Path, trj: nc.Dataset, state: int) -> mda.Universe:
    """
    Construct an MDAnalysis Universe from a MultiState NetCDF trajectory
    and apply standard analysis transformations.

    The Universe is created using the custom ``FEReader`` to extract a
    single state from a multistate simulation.

    Parameters
    ----------
    top : pathlib.Path or Topology
        Path to a topology file (e.g. PDB) or an already-loaded MDAnalysis
        topology object.
    trj : nc.Dataset
        Open NetCDF dataset produced by
        ``openmmtools.multistate.MultiStateReporter``.
    state : int
        Thermodynamic state index to extract from the multistate trajectory.

    Returns
    -------
    MDAnalysis.Universe
        A Universe with trajectory transformations applied.

    Notes
    -----
    Identifies two AtomGroups:

    - Protein, defined as having standard amino acid names, then filtered down to CA
    - Ligand, defined as "resname UNK"

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
    """
    u = mda.Universe(
        top,
        trj,
        index=state,
        index_method="state",
        format=FEReader,
    )
    prot = u.select_atoms("protein and name CA")
    ligand = u.select_atoms("resname UNK")

    if prot:
        # Unwrap all atoms
        unwrap_tr = unwrap(prot + ligand)

        # Shift chains + ligand
        chains = [seg.atoms for seg in prot.segments]
        shift = ClosestImageShift(chains[0], [*chains[1:], ligand])

        align = Aligner(prot)

        u.trajectory.add_transformations(
            unwrap_tr,
            shift,
            align,
        )
    else:
        # if there's no protein
        # - make the ligand not jump periodic images between frames
        # - align the ligand to minimize its RMSD
        nope = NoJump(ligand)
        align = Aligner(ligand)

        u.trajectory.add_transformations(
            nope,
            align,
        )

    return u


class Protein2DRMSD(AnalysisBase):
    """
    Flattened 2D RMSD matrix

    For all unique frame pairs ``(i, j)`` with ``i < j``, this function
    computes the RMSD between atomic coordinates after optimal alignment.
    Alignment is performed by centering each frame on its center of geometry,
    followed by rotational and translational superposition using the QCP method.

    Parameters
    ----------
    atomgroup: mda.AtomGroup
      Protein atoms (e.g. CA selection)
    weights: np.ndarray, optional
      Per-atom weights to use in the RMSD calculation. If ``None``,
      all atoms are weighted equally.

    Notes
    -----
    All atom positions are accumulated in memory during the trajectory
    iteration. For long trajectories or large systems this may result in
    significant memory usage. Consider using the ``step`` argument to
    ``run()`` to reduce the number of frames analyzed.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(self, atomgroup: mda.AtomGroup, weights: Optional[np.ndarray] = None, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._weights = weights
        self._ag = atomgroup

    def _prepare(self) -> None:
        self._coords = np.zeros((self.n_frames, self._ag.n_atoms, 3), dtype=np.float64)

    def _single_frame(self) -> None:
        self._coords[self._frame_index] = self._ag.positions

    def _conclude(self) -> None:
        nframes = self._coords.shape[0]

        # Pre-allocate numpy arrays
        n_pairs = nframes * (nframes - 1) // 2
        self.results.rmsd2d = np.empty(n_pairs)

        for idx, (i, j) in enumerate(itertools.combinations(range(nframes), 2)):
            posi, posj = self._coords[i], self._coords[j]
            self.results.rmsd2d[idx] = rms.rmsd(
                posi,
                posj,
                self._weights,
                center=True,
                superposition=True,
            )


class RMSDAnalysis(AnalysisBase):
    """
    1D RMSD time series for an AtomGroup.

    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup
      Atoms to compute RMSD for.
    reference: Optional[MDAnalysis.AtomGroup]
      Reference AtomGroup. If ``None``, the reference positions are taken
      from the first analyzed frame, so ``run(start=10)`` measures RMSD
      relative to frame 10, not frame 0.
    mass_weighted : bool, optional
      If True, compute mass-weighted RMSD.
    center : bool, optional
      If ``True``, subtract the center of geometry before computing RMSD.
      Defaults to ``False`` as the trajectory is assumed to be pre-centered.
    superposition : bool, optional
      If ``True``, perform rotational superposition before computing RMSD.
      Defaults to ``False`` as the trajectory is assumed to be pre-superposed.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        reference: Optional[mda.AtomGroup] = None,
        mass_weighted: bool = False,
        center: bool = False,
        superposition: bool = False,
        **kwargs,
    ):
        super().__init__(atomgroup.universe.trajectory, **kwargs)

        self._ag = atomgroup
        self._reference = reference if reference is not None else self._ag
        self._mass_weighted = mass_weighted
        self._center = center
        self._superposition = superposition

    def _prepare(self) -> None:
        self.results.rmsd = np.zeros(self.n_frames, dtype=np.float64)
        # reference is taken from the first analyzed frame, not necessarily frame 0
        self._reference_pos = self._reference.positions.copy()

        if self._mass_weighted:
            self._weights = self._ag.masses / np.mean(self._ag.masses)
        else:
            self._weights = None

    def _single_frame(self) -> None:
        self.results.rmsd[self._frame_index] = rms.rmsd(
            self._ag.positions,
            self._reference_pos,
            self._weights,
            center=self._center,
            superposition=self._superposition,
        )


class LigandCOMDrift(AnalysisBase):
    """
    Ligand center-of-mass displacement from initial position.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Ligand atoms for which the center-of-mass drift is calculated.

    Notes
    -----
    The reference COM is taken from whatever frame the trajectory is on when ``.run()`` is
    called, not necessarily the first analyzed frame. For consistent results,
    ensure the trajectory is at frame 0 (or your desired reference frame)
    before calling ``.run()``.

    PBC are not applied as the trajectory is assumed to have been
    pre-processed, ensuring the ligand does not jump between periodic images.
    Passing a box to apply the minimum image convention would give
    incorrect results for ligands that have drifted more than half a box
    length from their starting position.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(self, atomgroup: mda.AtomGroup, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup

    def _prepare(self) -> None:
        self.results.com_drift = np.zeros(self.n_frames, dtype=np.float64)
        # COM is captured from the current trajectory frame when .run() is called
        self._initial_com = self._ag.center_of_mass()

    def _single_frame(self) -> None:
        # no box argument, assumes the ligand stays in a consistent image;
        # applying the minimum image convention could mask large drifts > half a box length
        self.results.com_drift[self._frame_index] = mda.lib.distances.calc_bonds(
            self._ag.center_of_mass(),
            self._initial_com,
        )


def gather_rms_data(
    pdb_topology: pathlib.Path,
    dataset: pathlib.Path,
    skip: Optional[int] = None,
) -> dict[str, list[np.ndarray]]:
    """
    Compute structural RMSD-based metrics for a multistate BFE simulation.

    Parameters
    ----------
    pdb_topology : pathlib.Path
      Path to the PDB file defining system topology.
    dataset : pathlib.Path
      Path to the NetCDF trajectory file produced by a multistate simulation.
    skip : int, optional
      Frame stride for analysis. If ``None``, a stride is chosen such that
      approximately 500 frames are analyzed per state.

    Returns
    -------
    dict[str, list]
        Dictionary containing per-state analysis results with keys:
        ``protein_RMSD``, ``ligand_RMSD``, ``ligand_wander``,
        ``protein_2D_RMSD``, and ``time(ps)``.

    Notes
    -----
    For each thermodynamic state (lambda), this function:

    - Loads the trajectory using ``FEReader``
    - Applies standard PBC-handling and alignment transformations
    - Computes protein and ligand structural metrics over time

    The following analyses are produced per state:

    - 1D protein CA RMSD time series
    - 1D ligand RMSD time series
    - Ligand center-of-mass displacement from its initial position (``ligand_wander``)
    - Flattened 2D protein RMSD matrix (pairwise RMSD between frames)
    """
    output: dict[str, list[Any]] = {
        "protein_RMSD": [],
        "ligand_RMSD": [],
        "ligand_wander": [],
        "protein_2D_RMSD": [],
    }

    # Open the NetCDF file safely using a context manager
    with nc.Dataset(dataset) as ds:
        n_lambda = ds.dimensions["state"].size

        # If you're using a new multistate nc file, you need to account for
        # the position skip rate.
        if hasattr(ds, "PositionInterval"):
            n_frames = len(range(0, ds.dimensions["iteration"].size, ds.PositionInterval))
        else:
            n_frames = ds.dimensions["iteration"].size

        if skip is None:
            # find skip that would give ~500 frames of output
            # max against 1 to avoid skip=0 case
            skip = max(n_frames // 500, 1)

        u_top = mda.Universe(pdb_topology)

        for state_idx in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            u = make_Universe(u_top._topology, ds, state=state_idx)

            prot = u.select_atoms("protein and name CA")
            ligand = u.select_atoms("resname UNK")

            if prot:
                prot_rmsd = RMSDAnalysis(prot).run(step=skip)
                output["protein_RMSD"].append(prot_rmsd.results.rmsd)

                prot_rmsd2d = Protein2DRMSD(prot).run(step=skip)
                output["protein_2D_RMSD"].append(prot_rmsd2d.results.rmsd2d)

            if ligand:
                lig_rmsd = RMSDAnalysis(ligand, mass_weighted=True).run(step=skip)
                output["ligand_RMSD"].append(lig_rmsd.results.rmsd)

                lig_com_drift = LigandCOMDrift(ligand).run(step=skip)
                output["ligand_wander"].append(lig_com_drift.results.com_drift)

            output["time(ps)"] = np.arange(len(u.trajectory))[::skip] * u.trajectory.dt

    return output
