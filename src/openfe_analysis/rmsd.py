import itertools
import pathlib
from typing import Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import spyrmsd.rmsd as srmsd
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations import unwrap
from rdkit.Chem import rdmolops

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
    - protein, defined as having standard amino acid names, then filtered
      down to CA
    - ligand, defined as resname UNK

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
        # - align the ligand to minimise its RMSD
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


class SymmetryCorrectedLigandRMSD(AnalysisBase):
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
        super(SymmetryCorrectedLigandRMSD, self).__init__(atomgroup.universe.trajectory, **kwargs)

        self._ag = atomgroup
        self._mass_weighted = mass_weighted
        self._isomorphisms = None
        vdwradii = {
            "Cl": 1.75,
            "CL": 1.75,
            "Br": 1.85,
            "BR": 1.85,
            "Na": 2.27,
            "NA": 2.27,
        }
        atomgroup.guess_bonds(vdwradii)
        self._mol = atomgroup.convert_to("RDKIT")
        self._aprops = np.array([atom.GetAtomicNum() for atom in self._mol.GetAtoms()])
        self._am = rdmolops.GetAdjacencyMatrix(self._mol)

    def _prepare(self):
        self.results.rmsd = []
        self._reference = self._ag.positions
        self._ref_aprops = self._aprops
        self._ref_am = self._am

        if self._mass_weighted:
            self._weights = self._ag.masses / np.mean(self._ag.masses)
        else:
            self._weights = None

    def _single_frame(self):
        coords = self._ag.positions.copy()
        rmsd, isomorphisms, _ = srmsd._rmsd_isomorphic_core(
            coords1=coords,
            coords2=self._reference,
            aprops1=self._aprops,
            aprops2=self._ref_aprops,
            am1=self._am,
            am2=self._ref_am,
            center=False,
            minimize=False,
            isomorphisms=self._isomorphisms,
        )
        self.results.rmsd.append(rmsd)
        if self._isomorphisms is None:
            self._isomorphisms = isomorphisms

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
            u = make_Universe(u_top._topology, ds, state=i)
            bfactor = 0.25
            state_atoms = np.array([atom.ix for atom in u.atoms if atom.bfactor in (bfactor, 0.5)])
            state = u.atoms[state_atoms]

            prot = u.select_atoms("protein and name CA")
            ligand = u.select_atoms("resname UNK")
            state_lig = state.select_atoms("resname UNK")

            if prot:
                prot_rmsd = RMSDAnalysis(prot).run(step=skip)
                output["protein_RMSD"].append(prot_rmsd.results.rmsd)
                # prot_rmsd = rms.RMSD(prot).run(step=skip)
                # output["protein_RMSD"].append(prot_rmsd.results.rmsd.T[2])
                prot_rmsd2d = Protein2DRMSD(prot).run(step=skip)
                output["protein_2D_RMSD"].append(prot_rmsd2d.results.rmsd2d)

            if ligand:
                # lig_rmsd = RMSDAnalysis(ligand, mass_weighted=True).run(step=skip)
                lig_rmsd = SymmetryCorrectedLigandRMSD(state_lig, mass_weighted=True).run(step=skip)
                output["ligand_RMSD"].append(lig_rmsd.results.rmsd)
                # weight = ligand.masses / np.mean(ligand.masses)
                # lig_rmsd = rms.RMSD(ligand, weights=weight).run(step=skip)
                # output["ligand_RMSD"].append(lig_rmsd.results.rmsd.T[2])
                lig_com_drift = LigandCOMDrift(ligand).run(step=skip)
                output["ligand_wander"].append(lig_com_drift.results.com_drift)

            output["time(ps)"] = np.arange(len(u.trajectory))[::skip] * u.trajectory.dt

    return output
