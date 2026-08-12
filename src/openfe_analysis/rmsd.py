import itertools
import pathlib
from typing import Any, Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import spyrmsd.rmsd as srmsd
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.base import AnalysisBase
from rdkit import Chem

from .utils.apply_transformations import (
    apply_complex_alignment_transformations,
    apply_ligand_alignment_transformations,
)
from .utils.universe_utils import create_universe_single_state


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


class SymmetryCorrectedLigandRMSD(AnalysisBase):
    """
    Symmetry-corrected 1D RMSD time series for a ligand AtomGroup.

    Parameters
    ----------
    atomgroup : mda.AtomGroup
        Ligand atoms to compute RMSD for. If ``rdmol`` is not provided,
        bonds must be guessed on the atomgroup before instantiating this
        class; use :func:`guess_ligand_bonds` for this purpose.
    rdmol : Chem.Mol, optional
        RDKit molecule corresponding to ``atomgroup``. If provided, it is
        used directly and ``guess_ligand_bonds`` does not need to be called.
        If ``None``, the RDKit molecule is derived from ``atomgroup`` via
        ``convert_to("RDKIT")``.

    Raises
    ------
    ValueError
        If ``rdmol`` is ``None`` and no bonds are found on the atomgroup.
    ValueError
        If the number of atoms in ``atomgroup`` and ``rdmol`` do not match.
    """

    _analysis_algorithm_is_parallelizable = False

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        rdmol: Optional[Chem.Mol] = None,
        **kwargs,
    ):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        if rdmol is None:
            try:
                has_bonds = len(atomgroup.bonds) > 0
            except mda.exceptions.NoDataError:
                has_bonds = False
            if not has_bonds:
                raise ValueError(
                    "No bonds found on atomgroup. Call guess_ligand_bonds() "
                    "before instantiating SymmetryCorrectedLigandRMSD, or "
                    "pass an rdmol directly."
                )
        else:
            if len(atomgroup) != rdmol.GetNumAtoms():
                raise ValueError(
                    f"atomgroup has {len(atomgroup)} atoms but rdmol has "
                    f"{rdmol.GetNumAtoms()} atoms."
                )
        self._mol = rdmol if rdmol is not None else atomgroup.convert_to("RDKIT")
        self._aprops = np.array([atom.GetAtomicNum() for atom in self._mol.GetAtoms()])
        self._am = Chem.rdmolops.GetAdjacencyMatrix(self._mol)

    def _prepare(self):
        self.results.rmsd = np.zeros(self.n_frames, dtype=np.float64)
        # reference is taken from the first analyzed frame, not necessarily frame 0
        self._reference = self._ag.positions.copy()
        self._isomorphisms: list | None = None

    def _single_frame(self) -> None:
        frame_rmsd, isomorphisms, _ = srmsd._rmsd_isomorphic_core(
            coords1=self._ag.positions.copy(),
            coords2=self._reference,
            aprops1=self._aprops,
            aprops2=self._aprops,
            am1=self._am,
            am2=self._am,
            center=False,
            minimize=False,
            isomorphisms=self._isomorphisms,
        )
        self.results.rmsd[self._frame_index] = frame_rmsd
        # cache isomorphisms after first frame to avoid redundant graph matching
        if self._isomorphisms is None:
            self._isomorphisms = isomorphisms


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
    protein_selection: str = "protein and name CA",
    ligand_selection: str = "resname UNK",
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
    protein_selection : str
      MDAnalysis selection string for the protein atoms used in RMSD
      calculations. Default is ``"protein and name CA"``.
    ligand_selection : str
      MDAnalysis selection string for the ligand atoms. Default is
      ``"resname UNK"``.

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
        u_top.select_atoms("protein").guess_bonds()

        for state_idx in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            universe = create_universe_single_state(u_top._topology, ds, state_idx)
            prot = universe.select_atoms(protein_selection)
            ligand = universe.select_atoms(ligand_selection)

            if prot:
                apply_complex_alignment_transformations(
                    universe,
                    protein=prot,
                    ligands=[ligand] if ligand.n_atoms > 0 else None,
                )

            elif ligand.n_atoms > 0:
                apply_ligand_alignment_transformations(universe, ligand=ligand)

            output["time(ps)"] = (
                np.arange(len(universe.trajectory))[::skip] * universe.trajectory.dt
            )
            # unwrap/shift/align run once per frame, here
            universe.transfer_to_memory(step=skip)

            if prot:
                prot_rmsd = RMSDAnalysis(prot).run()
                output["protein_RMSD"].append(prot_rmsd.results.rmsd)

                prot_rmsd2d = Protein2DRMSD(prot).run()
                output["protein_2D_RMSD"].append(prot_rmsd2d.results.rmsd2d)

            if ligand:
                lig_rmsd = RMSDAnalysis(ligand, mass_weighted=True).run()
                output["ligand_RMSD"].append(lig_rmsd.results.rmsd)

                lig_com_drift = LigandCOMDrift(ligand).run()
                output["ligand_wander"].append(lig_com_drift.results.com_drift)

    return output
