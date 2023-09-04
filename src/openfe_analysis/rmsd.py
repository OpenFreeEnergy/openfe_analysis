import MDAnalysis as mda
from MDAnalysis.analysis import rms
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
    """
    Produces:
    - ligand RMSD
    - protein RMSD

    For each constant state/lambda window
    """
    output = {
        'protein_RMSD': [],
        'ligand_RMSD': [],
        'ligand_wander': [],
    }

    n_lambda = 11  # detect number of lambda windows
    for i in range(n_lambda):
        u = make_Universe(pdb_topology, dataset, state=i)

        # todo: apply centering transformation to universe

        prot = u.select_atoms('protein and name CA')
        if prot:
            rmsd = generate_rmsd(prot)

            output['protein_RMSD'].append(rmsd)

        ligand = u.select_atoms('resname UNK')
        if ligand:
            rmsd = generate_rmsd(ligand)

            output['ligand_RMSD'].append(rmsd)

            output['ligand_wander'].append(ligand_wander(ligand))

    return output


def generate_rmsd(ag: mda.AtomGroup) -> list[float]:
    """Returns the RMSD for ag over the trajectory"""
    p1 = ag.positions
    w = ag.masses / np.mean(ag.masses)

    output = []
    for ts in ag.universe.trajectory:
        # this rmsd call wouldn't usually work
        # except for the trajectory transform we have going on
        # for the protein case, we have already aligned
        # for the ligand case, it maintains the protein alignment
        output.append(
            rms.rmsd(ag.positions, p1, weights=w,
                     center=False, superposition=False
                     )
        )

    return output


def ligand_wander(ag) -> list[float]:
    """Tracks motion of ligand over time

    Returns list of displacement relative to start point

    Note: this is done on the centered trajectory, so it's a little off
    """
    start = ag.center_of_mass()

    output = []
    for ts in ag.universe.trajectory:
        output.append(
            mda.lib.distances.calc_bonds(start, ag.center_of_mass())
        )

    return output