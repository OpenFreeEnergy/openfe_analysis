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


def make_Universe(
    top: pathlib.Path,
    trj: nc.Dataset,
    state: int,
    ligand_resname: str = "UNK",
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
    ligand_resname : str, default 'UNK'
      Residue name(s) for ligands. Supports multiple ligands.
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
    prot = u.select_atoms(protein_selection)
    ligands = [res.atoms for res in u.residues if res.resname == ligand_resname]

    if prot:
        # Unwrap all atoms
        unwrap_tr = unwrap(prot)

        # Shift chains + ligand
        chains = [seg.atoms for seg in prot.segments]
        shift = ClosestImageShift(chains[0], [*chains[1:], *ligands])
        # Make each protein chain whole
        for frag in prot.fragments:
            make_whole(frag, reference_atom=frag[0])

        align = Aligner(prot)

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


def gather_rms_data(
    pdb_topology: pathlib.Path,
    dataset: pathlib.Path,
    skip: Optional[int] = None,
    ligand_resname: str = "UNK",
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
    ligand_resname : str, default 'UNK'
      Residue name for ligand(s). Supports multiple ligands.
    protein_selection : str, default 'protein and name CA'
      MDAnalysis selection string for the protein atoms to consider.

    Produces, for each lambda state:
    - 1D protein RMSD timeseries 'protein_RMSD'
    - ligand RMSD timeseries
    - ligand COM motion 'ligand_wander'
    - 2D protein RMSD plot
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

        pb = tqdm.tqdm(total=int(n_frames / skip) * n_lambda)

        u_top = mda.Universe(pdb_topology)

        for i in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            u = make_Universe(
                u_top._topology,
                ds,
                state=i,
                ligand_resname=ligand_resname,
                protein_selection=protein_selection,
            )

            prot = u.select_atoms(protein_selection)
            ligands = [res.atoms for res in u.residues if res.resname == ligand_resname]

            # Prepare storage
            if prot:
                prot_positions = np.empty(
                    (len(u.trajectory[::skip]), len(prot), 3), dtype=np.float32
                )
                prot_start = prot.positions.copy()
                prot_rmsd = []

            lig_starts = [lig.positions.copy() for lig in ligands]
            lig_initial_coms = [lig.center_of_mass() for lig in ligands]
            lig_rmsd: list[list[float]] = [[] for _ in ligands]
            lig_wander: list[list[float]] = [[] for _ in ligands]

            for ts_i, ts in enumerate(u.trajectory[::skip]):
                pb.update()
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
                    lig_wander[i].append(
                        # distance between start and current ligand position
                        # ignores PBC, but we've already centered the traj
                        mda.lib.distances.calc_bonds(lig.center_of_mass(), lig_initial_coms[i])
                    )

            if prot:
                # can ignore weights here as it's all Ca
                rmsd2d = twoD_RMSD(prot_positions, w=None)  # prot_weights)
                output["protein_RMSD"].append(prot_rmsd)
                output["protein_2D_RMSD"].append(rmsd2d)
            if ligands:
                output["ligand_RMSD"].append(lig_rmsd)
                output["ligand_wander"].append(lig_wander)

            output["time(ps)"] = list(np.arange(len(u.trajectory))[::skip] * u.trajectory.dt)

    return output


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
      a flattened version of the 2d
    """
    nframes, _, _ = positions.shape

    output = []

    for i, j in itertools.combinations(range(nframes), 2):
        posi, posj = positions[i], positions[j]

        rmsd = rms.rmsd(posi, posj, w, center=True, superposition=True)

        output.append(rmsd)

    return output
