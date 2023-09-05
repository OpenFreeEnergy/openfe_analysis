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

    n_lambda = 11  # todo: detect number of lambda windows
    for i in range(n_lambda):
        u = make_Universe(pdb_topology, dataset, state=i)

        prot = u.select_atoms('protein and name CA')
        ligand = u.select_atoms('resname UNK')

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
                    mda.lib.distances.calc_bonds(ligand.center_of_mass, ligand_initial_com)
                )

        if prot:
            output['protein_RMSD'].append(this_protein_rmsd)
        if ligand:
            output['ligand_RMSD'].append(this_ligand_rmsd)
            output['ligand_wander'].append(this_ligand_wander)

    output['time(ps)'] = list(np.arange(len(u.trajectory)) * u.trajectory.dt)

    return output
