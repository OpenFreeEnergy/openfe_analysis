import itertools
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import netCDF4 as nc
import numpy as np

from .reader import FEReader
from .transformations import (
    NoJump, Minimiser, Aligner
)


def make_Universe(top, trj, state):
    u = mda.Universe(
        top, trj, state_id=state,
        format='openfe RFE',
    )
    prot = u.select_atoms('protein and name CA')
    ligand = u.select_atoms('resname UNK')

    nope = NoJump(prot)
    minnie = Minimiser(prot, ligand)
    align = Aligner(prot)

    u.trajectory.add_transformations(
        nope, minnie, align,
    )

    return u


def gather_rms_data(pdb_topology, dataset):
    """Generate structural analysis of RBFE simulation

    Parameters
    ----------
    pdb_topology : pathlib.Path
      path to pdb topology
    dataset : pathlib.Path
      path to nc trajectory

    Produces:
    - protein RMSD
    - ligand RMSD
    - ligand COM motion

    For ligand metrics, each frame is first aligned to minimise the protein
    RMSD.

    For each constant state/lambda window
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
        prot_weights = prot.masses / np.mean(prot.masses)
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
                    rms.rmsd(prot.positions, prot_start, prot_weights,
                             center=False, superposition=False)
                )
            if ligand:
                this_ligand_rmsd.append(
                    rms.rmsd(ligand.positions, ligand_start, ligand_weights,
                             center=False, superposition=False)
                )
                this_ligand_wander.append(
                    mda.lib.distances.calc_bonds(ligand.center_of_mass(), ligand_initial_com)
                )

        if prot:
            rmsd2d = twoD_RMSD(prot2d, prot_weights)
            output['protein_RMSD'].append(this_protein_rmsd)
            output['protein_2D_RMSD'].append(rmsd2d)
        if ligand:
            output['ligand_RMSD'].append(this_ligand_rmsd)
            output['ligand_wander'].append(this_ligand_wander)

    output['time(ps)'] = list(np.arange(len(u.trajectory)) * u.trajectory.dt)

    return output


def twoD_RMSD(positions, w) -> list[float]:
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
