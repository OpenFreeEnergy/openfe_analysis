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


def make_Universe(top: pathlib.Path, trj: nc.Dataset, state: int) -> mda.Universe:
    """
    Construct an MDAnalysis Universe from a MultiState NetCDF trajectory
    and apply standard analysis transformations.

    The Universe is created using the custom ``FEReader`` to extract a
    single state from a multistate simulation.

    Identifies two AtomGroups:
    - Protein, defined as having standard amino acid names, then filtered down to CA
    - Ligand, defined as resname UNK

    Depending on whether a protein is present, a sequence of trajectory
    transformations is applied:

    If a protein is present:
    - Prevents the protein from jumping between periodic images (class:`NoJump`)
    - Moves the ligand to the image closest to the protein (:class:`Minimiser`)
    - Aligns the entire system to minimise the protein RMSD (:class:`Aligner`)

    If only a ligand is present:
    - Prevents the ligand from jumping between periodic images
    - Aligns the ligand to minimize its RMSD

    Parameters
    ----------
    top : pathlib.Path or Topology
        Path to a topology file (e.g. PDB) or an already-loaded MDAnalysis
        topology object.
    trj : netCDF4.Dataset
        Open NetCDF dataset produced by
        ``openmmtools.multistate.MultiStateReporter``.
    state : int
        Thermodynamic state index to extract from the multistate trajectory.

    Returns
    -------
    MDAnalysis.Universe
        A Universe with trajectory transformations applied.
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
            u = make_Universe(u_top._topology, ds, state=i)

            prot = u.select_atoms("protein and name CA")
            ligand = u.select_atoms("resname UNK")

            # save coordinates for 2D RMSD matrix
            # TODO: Some smart guard to avoid allocating a silly amount of memory?
            prot2d = np.empty((len(u.trajectory[::skip]), len(prot), 3), dtype=np.float32)

            prot_start = prot.positions
            ligand_start = ligand.positions
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
