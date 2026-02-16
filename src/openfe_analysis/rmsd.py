import itertools
import pathlib
from typing import Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import tqdm
from MDAnalysis.analysis import rms
from MDAnalysis.lib.mdamath import make_whole
from MDAnalysis.transformations import unwrap
from numpy import typing as npt

from .reader import FEReader
from .transformations import Aligner, ClosestImageShift, NoJump


def _select_protein_and_ligands(
    u: mda.Universe,
    protein_selection: str,
    ligand_selection: str,
) -> tuple[mda.core.groups.AtomGroup, list[mda.core.groups.AtomGroup]]:
    protein = u.select_atoms(protein_selection)
    lig_atoms = u.select_atoms(ligand_selection)
    # split into individual ligands by residue
    ligands = [res.atoms for res in lig_atoms.residues]

    return protein, ligands


def make_Universe(
    top: pathlib.Path,
    trj: nc.Dataset,
    state: int,
    ligand_selection: str = "resname UNK",
    protein_selection: str = "protein and name CA",
) -> mda.Universe:
    """
    Creates a Universe and applies transformations for protein and ligands.

    Parameters
    ----------
    top : pathlib.Path
      Path to the topology file.
    trj : nc.Dataset
      Trajectory dataset.
    state : int
      State index in the trajectory.
    ligand_selection : str, default 'resname UNK'
      MDAnalysis selection string for ligands. Supports multiple ligands.
    protein_selection : str, default 'protein and name CA'
      MDAnalysis selection string for the protein atoms to consider.

    Returns
    -------
    mda.Universe
      Universe with transformations applied.

    Notes
    -----
    If a protein is present:
    - prevents the protein from jumping between periodic images
    - moves the ligand to the image closest to the protein
    - aligns the entire system to minimise the protein RMSD

    If only a ligand is present:
    - prevents the ligand from jumping between periodic images
    """
    u = mda.Universe(
        top,
        trj,
        state_id=state,
        format=FEReader,
    )

    protein, ligands = _select_protein_and_ligands(u, protein_selection, ligand_selection)

    if protein:
        # Unwrap all atoms
        unwrap_tr = unwrap(protein + ligands)

        # Shift chains + ligand
        chains = [seg.atoms for seg in protein.segments]
        shift = ClosestImageShift(chains[0], [*chains[1:], *ligands])

        align = Aligner(protein)

        u.trajectory.add_transformations(
            unwrap_tr,
            shift,
            align,
        )
    else:
        # if there's no protein
        # - make the ligands not jump periodic images between frames
        # - align the ligands to minimise its RMSD
        for lig in ligands:
            u.trajectory.add_transformations(NoJump(lig), Aligner(lig))

    return u


def twoD_RMSD(positions: np.ndarray, w: Optional[npt.NDArray]) -> list[float]:
    """2 dimensions RMSD

    Parameters
    ----------
    positions : np.ndarray
      the protein positions for the entire trajectory
    w : np.ndarray, optional
      weights array

    Returns
    -------
    rmsd_matrix : list
      Flattened list of RMSD values between all frame pairs.
    """
    nframes, _, _ = positions.shape

    output = []

    for i, j in itertools.combinations(range(nframes), 2):
        posi, posj = positions[i], positions[j]

        rmsd = rms.rmsd(posi, posj, w, center=True, superposition=True)

        output.append(rmsd)

    return output


def analyze_state(
    u: mda.Universe,
    prot: Optional[mda.core.groups.AtomGroup],
    ligands: list[mda.core.groups.AtomGroup],
    skip: int,
) -> tuple[
    Optional[list[float]],
    Optional[np.ndarray],
    Optional[list[list[float]]],
    Optional[list[list[float]]],
]:
    """
    Compute RMSD and COM drift for a single lambda state.

    Parameters
    ----------
    u : mda.Universe
        Universe containing the trajectory.
    prot : AtomGroup or None
        Protein atoms to compute RMSD for.
    ligands : list of AtomGroups
        Ligands to compute RMSD and COM drift for.
    skip : int
        Step size to skip frames (e.g., every `skip`-th frame).

    Returns
    -------
    protein_rmsd : list[float] or None
        RMSD of protein per frame, if protein is present.
    protein_2D_rmsd : list[float] or None
        Flattened 2D RMSD between all protein frames.
    ligand_rmsd : list of list[float] or None
        RMSD of each ligand per frame.
    ligand_com_drift : list of list[float] or None
        COM drift of each ligand per frame.
    """
    # Prepare storage
    if prot:
        prot_positions = np.empty((len(u.trajectory[::skip]), len(prot), 3), dtype=np.float32)
        prot_start = prot.positions
        prot_rmsd = []
    else:
        prot_positions = None
        prot_rmsd = None

    lig_starts = [lig.positions for lig in ligands]
    lig_initial_coms = [lig.center_of_mass() for lig in ligands]
    lig_rmsd: list[list[float]] = [[] for _ in ligands]
    lig_com_drift: list[list[float]] = [[] for _ in ligands]

    for ts_i, ts in enumerate(u.trajectory[::skip]):
        if prot:
            prot_positions[ts_i, :, :] = prot.positions
            prot_rmsd.append(
                rms.rmsd(
                    prot.positions,
                    prot_start,
                    None,  # prot_weights,
                    center=False,
                    superposition=False,
                )
            )
        for i, lig in enumerate(ligands):
            lig_rmsd[i].append(
                rms.rmsd(
                    lig.positions,
                    lig_starts[i],
                    lig.masses / np.mean(lig.masses),
                    center=False,
                    superposition=False,
                )
            )
            lig_com_drift[i].append(
                # distance between start and current ligand position
                # ignores PBC, but we've already centered the traj
                mda.lib.distances.calc_bonds(lig.center_of_mass(), lig_initial_coms[i])
            )

    if prot:
        # can ignore weights here as it's all Ca
        rmsd2d = twoD_RMSD(prot_positions, w=None)  # prot_weights)

    return prot_rmsd, rmsd2d, lig_rmsd, lig_com_drift


def gather_rms_data(
    pdb_topology: pathlib.Path,
    dataset: pathlib.Path,
    skip: Optional[int] = None,
    ligand_selection: str = "resname UNK",
    protein_selection: str = "protein and name CA",
) -> dict[str, list[float]]:
    """Generate structural analysis of RBFE simulation

    Parameters
    ----------
    pdb_topology : pathlib.Path
      path to pdb topology
    dataset : pathlib.Path
      path to nc trajectory
    skip : int, optional
      step at which to progress through the trajectory.  by default, selects a
      step that produces roughly 500 frames of analysis per replicate
    ligand_selection : str, optional
        MDAnalysis selection string for ligands (default "resname UNK").
    protein_selection : str, optional
        MDAnalysis selection string for protein (default "protein and name CA").

    Returns
    -------
    output : dict[str, list]
        Dictionary containing:
        - 'protein_RMSD': list of protein RMSD per state
        - 'protein_2D_RMSD': list of 2D RMSD per state
        - 'ligand_RMSD': list of ligand RMSD per state
        - 'ligand_COM_drift': list of ligand COM drift per state
    """
    output = {
        "protein_RMSD": [],
        "ligand_RMSD": [],
        "ligand_COM_drift": [],
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

        pb = tqdm.tqdm(total=int(n_frames / skip) * n_lambda)

        u_top = mda.Universe(pdb_topology)

        for state in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            u = make_Universe(
                u_top._topology,
                ds,
                state=state,
                ligand_selection=ligand_selection,
                protein_selection=protein_selection,
            )
            prot, ligands = _select_protein_and_ligands(u, protein_selection, ligand_selection)
            prot_rmsd, rmsd2d, lig_rmsd, lig_com_drift = analyze_state(u, prot, ligands, skip)

            if prot:
                output["protein_RMSD"].append(prot_rmsd)
                output["protein_2D_RMSD"].append(rmsd2d)

            if ligands:
                output["ligand_RMSD"].append(lig_rmsd)
                output["ligand_COM_drift"].append(lig_com_drift)

            output["time(ps)"] = list(np.arange(len(u.trajectory))[::skip] * u.trajectory.dt)
            pb.update(len(u.trajectory[::skip]))

    return output
