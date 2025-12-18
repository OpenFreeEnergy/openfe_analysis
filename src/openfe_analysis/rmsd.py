import itertools
import pathlib
from typing import Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import tqdm
from MDAnalysis.analysis import rms
from MDAnalysis.transformations import make_whole, unwrap
from numpy import typing as npt

from .reader import FEReader
from .transformations import Aligner, Minimiser, NoJump


def make_Universe(top: pathlib.Path, trj: nc.Dataset, state: int) -> mda.Universe:
    """Makes a Universe and applies some transformations

    Identifies two AtomGroups:
    - protein, defined as having standard amino acid names, then filtered
      down to CA
    - ligand, defined as resname UNK

    Then applies some transformations.

    If a protein is present:
    - prevents the protein from jumping between periodic images
    - moves the ligand to the image closest to the protein
    - aligns the entire system to minimise the protein RMSD

    If only a ligand:
    - prevents the ligand from jumping between periodic images
    """
    u = mda.Universe(
        top,
        trj,
        state_id=state,
        format=FEReader,
    )
    prot = u.select_atoms("protein and name CA")
    ligand = u.select_atoms("resname UNK")

    if prot:
        # if there's a protein in the system:
        # - make the protein whole across periodic images between frames
        # - put the ligand in the closest periodic image as the protein
        # - align everything to minimise protein RMSD
        make_whole_tr = make_whole(prot, compound="segments")
        unwrap_tr = unwrap(prot)
        minnie = Minimiser(prot, ligand)
        align = Aligner(prot)

        u.trajectory.add_transformations(
            make_whole_tr,
            unwrap_tr,
            minnie,
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


def gather_rms_data(
    pdb_topology: pathlib.Path, dataset: pathlib.Path, skip: Optional[int] = None
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

    ds = nc.Dataset(dataset)
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
        u = make_Universe(u_top._topology, ds, state=i)

        prot = u.select_atoms("protein and name CA")
        ligand = u.select_atoms("resname UNK")

        # save coordinates for 2D RMSD matrix
        # TODO: Some smart guard to avoid allocating a silly amount of memory?
        prot2d = np.empty((len(u.trajectory[::skip]), len(prot), 3), dtype=np.float32)

        # Would this copy be safer?
        prot_start = prot.positions.copy()
        ligand_start = ligand.positions.copy()
        ligand_initial_com = ligand.center_of_mass()
        ligand_weights = ligand.masses / np.mean(ligand.masses)

        this_protein_rmsd = []
        this_ligand_rmsd = []
        this_ligand_wander = []

        for ts_i, ts in enumerate(u.trajectory[::skip]):
            pb.update()

            if prot:
                prot2d[ts_i, :, :] = prot.positions
                this_protein_rmsd.append(
                    rms.rmsd(
                        prot.positions,
                        prot_start,
                        None,  # prot_weights,
                        center=False,
                        superposition=False,
                    )
                )
            if ligand:
                this_ligand_rmsd.append(
                    rms.rmsd(
                        ligand.positions,
                        ligand_start,
                        ligand_weights,
                        center=False,
                        superposition=False,
                    )
                )
                this_ligand_wander.append(
                    # distance between start and current ligand position
                    # ignores PBC, but we've already centered the traj
                    mda.lib.distances.calc_bonds(ligand.center_of_mass(), ligand_initial_com)
                )

        if prot:
            # can ignore weights here as it's all Ca
            rmsd2d = twoD_RMSD(prot2d, w=None)  # prot_weights)
            output["protein_RMSD"].append(this_protein_rmsd)
            output["protein_2D_RMSD"].append(rmsd2d)
        if ligand:
            output["ligand_RMSD"].append(this_ligand_rmsd)
            output["ligand_wander"].append(this_ligand_wander)

        output["time(ps)"] = list(np.arange(len(u.trajectory))[::skip] * u.trajectory.dt)

    return output


def twoD_RMSD(positions, w: Optional[npt.NDArray]) -> list[float]:
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
