import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import netCDF4 as nc
import numpy as np
from numpy import typing as npt
import pathlib
from typing import Optional

from .reader import FEReader
from .transformations import (
    NoJump, Minimiser, Aligner
)


def make_Universe(top: pathlib.Path,
                  trj: nc.Dataset,
                  state: int) -> mda.Universe:
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
        top, trj, state_id=state,
        format=FEReader,
    )
    prot = u.select_atoms('protein and name CA')
    ligand = u.select_atoms('resname UNK')

    if prot:
        # if there's a protein in the system:
        # - make the protein not jump periodic images between frames
        # - put the ligand in the closest periodic image as the protein
        # - align everything to minimise protein RMSD
        nope = NoJump(prot)
        minnie = Minimiser(prot, ligand)
        align = Aligner(prot)

        u.trajectory.add_transformations(
            nope, minnie, align,
        )
    else:
        # if there's no protein
        # - make the ligand not jump periodic images between frames
        # - align the ligand to minimise its RMSD
        nope = NoJump(ligand)
        align = Aligner(ligand)

        u.trajectory.add_transformations(
            nope, align,
        )

    return u


def gather_rms_data(pdb_topology: pathlib.Path,
                    dataset: pathlib.Path) -> dict[str, list[float]]:
    """Generate structural analysis of RBFE simulation

    Parameters
    ----------
    pdb_topology : pathlib.Path
      path to pdb topology
    dataset : pathlib.Path
      path to nc trajectory

    Produces, for each lambda state:
    - 1D protein RMSD timeseries 'protein_RMSD'
    - ligand RMSD timeseries
    - ligand COM motion 'ligand_wander'
    - 2D protein RMSD plot
    """
    output = {
        'protein_RMSD': [],
        'ligand_RMSD': [],
        'ligand_wander': [],
        'protein_2D_RMSD': [],
    }

    ds = nc.Dataset(dataset)
    n_lambda = ds.dimensions['state'].size
    for i in range(n_lambda):
        u = make_Universe(pdb_topology, ds, state=i)

        prot = u.select_atoms('protein and name CA')
        ligand = u.select_atoms('resname UNK')

        # save coordinates for 2D RMSD matrix
        # TODO: Some smart guard to avoid allocating a silly amount of memory?
        prot2d = np.empty((len(u.trajectory), len(prot), 3), dtype=np.float32)

        prot_start = prot.positions
        # prot_weights = prot.masses / np.mean(prot.masses)
        ligand_start = ligand.positions
        ligand_initial_com = ligand.center_of_mass()
        ligand_weights = ligand.masses / np.mean(ligand.masses)

        this_protein_rmsd = []
        this_ligand_rmsd = []
        this_ligand_wander = []

        for ts in u.trajectory:
            if prot:
                prot2d[ts.frame, :, :] = prot.positions
                this_protein_rmsd.append(
                    rms.rmsd(prot.positions, prot_start, None,  # prot_weights,
                             center=False, superposition=False)
                )
            if ligand:
                this_ligand_rmsd.append(
                    rms.rmsd(ligand.positions, ligand_start, ligand_weights,
                             center=False, superposition=False)
                )
                this_ligand_wander.append(
                    # distance between start and current ligand position
                    # ignores PBC, but we've already centered the traj
                    mda.lib.distances.calc_bonds(ligand.center_of_mass(),
                                                 ligand_initial_com)
                )

        if prot:
            rmsd2d = twoD_RMSD(prot2d, None)  # prot_weights)
            output['protein_RMSD'].append(this_protein_rmsd)
            output['protein_2D_RMSD'].append(rmsd2d)
        if ligand:
            output['ligand_RMSD'].append(this_ligand_rmsd)
            output['ligand_wander'].append(this_ligand_wander)

        output['time(ps)'] = list(np.arange(len(u.trajectory)) * u.trajectory.dt)

    return output


def twoD_RMSD(positions, w: Optional[npt.NDArray]) -> list[float]:
    """2 dimensions RMSD

    Parameters
    ----------
    positions : np.ndarray
      the protein positions for the entire trajectory
    w : np.ndarray
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

        rmsd = rms.rmsd(posi, posj, w,
                        center=True, superposition=True)

        output.append(rmsd)

    return output
