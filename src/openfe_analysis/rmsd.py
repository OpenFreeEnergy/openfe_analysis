import itertools
import pathlib
from typing import Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.base import AnalysisBase

from .reader import FEReader
from .utils.universe_transformations import apply_transformations, create_universe


class Protein2DRMSD(AnalysisBase):
    """
    Flattened 2D RMSD matrix

    For all unique frame pairs ``(i, j)`` with ``i < j``, this function
    computes the RMSD between atomic coordinates after optimal alignment.
    """

    def __init__(self, atomgroup, weights=None, **kwargs):
        """
        Parameters
        ----------
        atomgroup: AtomGroup
          Protein atoms (e.g. CA selection)
        weights: np.ndarray, optional
          Per-atom weights to use in the RMSD calculation. If ``None``,
          all atoms are weighted equally.
        """
        super(Protein2DRMSD, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self._weights = weights
        self._ag = atomgroup

    def _prepare(self):
        self._coords = []
        self.results.rmsd2d = []

    def _single_frame(self):
        self._coords.append(self._ag.positions)

    def _conclude(self):
        positions = np.asarray(self._coords)
        nframes, _, _ = positions.shape

        output = []
        for i, j in itertools.combinations(range(nframes), 2):
            posi, posj = positions[i], positions[j]
            rmsd = rms.rmsd(
                posi,
                posj,
                self._weights,
                center=True,
                superposition=True,
            )
            output.append(rmsd)

        self.results.rmsd2d = np.asarray(output)


class RMSDAnalysis(AnalysisBase):
    """
    1D RMSD time series for an AtomGroup.

    Parameters
    ----------
    atomgroup : MDAnalysis.AtomGroup
      Atoms to compute RMSD for.
    mass_weighted : bool, optional
      If True, compute mass-weighted RMSD.
    """

    def __init__(self, atomgroup, mass_weighted=False, **kwargs):
        super(RMSDAnalysis, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self._ag = atomgroup
        self._mass_weighted = mass_weighted

    def _prepare(self):
        self.results.rmsd = []
        self._reference = self._ag.positions

        if self._mass_weighted:
            self._weights = self._ag.masses / np.mean(self._ag.masses)
        else:
            self._weights = None

    def _single_frame(self):
        rmsd = rms.rmsd(
            self._ag.positions,
            self._reference,
            self._weights,
            center=False,
            superposition=False,
        )
        self.results.rmsd.append(rmsd)

    def _conclude(self):
        self.results.rmsd = np.asarray(self.results.rmsd)


class LigandCOMDrift(AnalysisBase):
    """
    Ligand center-of-mass displacement from initial position.
    """

    def __init__(self, atomgroup, **kwargs):
        super(LigandCOMDrift, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self._ag = atomgroup

    def _prepare(self):
        self.results.com_drift = []
        self._initial_com = self._ag.center_of_mass()

    def _single_frame(self):
        # distance between start and current ligand position
        # ignores PBC, but we've already centered the traj
        drift = mda.lib.distances.calc_bonds(
            self._ag.center_of_mass(),
            self._initial_com,
        )
        self.results.com_drift.append(drift)

    def _conclude(self):
        self.results.com_drift = np.asarray(self.results.com_drift)


def gather_rms_data(
    pdb_topology: pathlib.Path, dataset: pathlib.Path, skip: Optional[int] = None
) -> dict[str, list[float]]:
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
      - Ligand center-of-mass displacement from its initial position
        (``ligand_wander``)
      - Flattened 2D protein RMSD matrix (pairwise RMSD between frames)
    """
    output = {
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

        for i in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            u = create_universe(u_top._topology, ds, i)
            prot = u.select_atoms("protein and name CA")
            ligand = u.select_atoms("resname UNK")

            apply_transformations(u, prot, ligand)

            if prot:
                prot_rmsd = RMSDAnalysis(prot).run(step=skip)
                output["protein_RMSD"].append(prot_rmsd.results.rmsd)
                # prot_rmsd = rms.RMSD(prot).run(step=skip)
                # output["protein_RMSD"].append(prot_rmsd.results.rmsd.T[2])
                prot_rmsd2d = Protein2DRMSD(prot).run(step=skip)
                output["protein_2D_RMSD"].append(prot_rmsd2d.results.rmsd2d)

            if ligand:
                lig_rmsd = RMSDAnalysis(ligand, mass_weighted=True).run(step=skip)
                output["ligand_RMSD"].append(lig_rmsd.results.rmsd)
                # weight = ligand.masses / np.mean(ligand.masses)
                # lig_rmsd = rms.RMSD(ligand, weights=weight).run(step=skip)
                # output["ligand_RMSD"].append(lig_rmsd.results.rmsd.T[2])
                lig_com_drift = LigandCOMDrift(ligand).run(step=skip)
                output["ligand_wander"].append(lig_com_drift.results.com_drift)

            output["time(ps)"] = np.arange(len(u.trajectory))[::skip] * u.trajectory.dt

    return output
