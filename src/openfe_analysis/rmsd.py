import itertools
import pathlib
from dataclasses import asdict, dataclass, field
from typing import List, Optional

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


@dataclass
class SingleLigandRMSData:
    rmsd: list[float]
    com_drift: list[float]
    resname: str
    resid: int
    segid: str


@dataclass
class LigandsRMSData:
    ligands: list[SingleLigandRMSData]

    def __iter__(self):
        return iter(self.ligands)

    def __len__(self):
        return len(self.ligands)

    def __getitem__(self, idx):
        return self.ligands[idx]


@dataclass
class StateRMSData:
    protein_rmsd: list[float] | None
    protein_2d_rmsd: list[float] | None
    ligands: LigandsRMSData | None


@dataclass
class RMSResults:
    time_ps: list[float]
    states: list[StateRMSData] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert results to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(
            time_ps=d["time_ps"],
            states=[
                StateRMSData(
                    protein_rmsd=s["protein_rmsd"],
                    protein_2d_rmsd=s["protein_2d_rmsd"],
                    ligands=(
                        LigandsRMSData(
                            ligands=[SingleLigandRMSData(**lig) for lig in s["ligands"]["ligands"]]
                        )
                        if s["ligands"] is not None
                        else None
                    ),
                )
                for s in d["states"]
            ],
        )


def select_protein_and_ligands(
    u: mda.Universe,
    protein_selection: str,
    ligand_selection: str,
):
    prot = u.select_atoms(protein_selection)

    lig_residues = u.select_atoms(ligand_selection).residues
    print([res.resid for res in lig_residues])
    print([res.segid for res in lig_residues])

    # split into individual ligands by residue
    ligands = [res.atoms for res in lig_residues]

    return prot, ligands


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

    prot, ligands = select_protein_and_ligands(u, protein_selection, ligand_selection)

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
    ligand_selection: str = "resname UNK",
    protein_selection: str = "protein and name CA",
) -> RMSResults:
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
    ligand_selection: str = "resname UNK",
      MDAnalysis selection string for ligand(s). Supports multiple ligands.
    protein_selection : str, default 'protein and name CA'
      MDAnalysis selection string for the protein atoms to consider.

    Returns
    -------
    RMSResults
        Per-state RMSD data for protein and ligands.
        Produces, for each lambda state:
        - 1D protein RMSD timeseries 'protein_RMSD'
        - ligand RMSD timeseries
        - ligand COM motion 'ligand_COM_drift'
        - 2D protein RMSD plot
    """
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

        states: list[StateRMSData] = []

        for i in range(n_lambda):
            # cheeky, but we can read the PDB topology once and reuse per universe
            # this then only hits the PDB file once for all replicas
            u = make_Universe(
                u_top._topology,
                ds,
                state=i,
                ligand_selection=ligand_selection,
                protein_selection=protein_selection,
            )

            prot, ligands = select_protein_and_ligands(u, protein_selection, ligand_selection)

            # Prepare storage
            if prot:
                prot_positions = np.empty(
                    (len(u.trajectory[::skip]), len(prot), 3), dtype=np.float32
                )
                prot_start = prot.positions.copy()
                prot_rmsd = []
            else:
                prot_rmsd = []
                prot_positions = None

            lig_starts = [lig.positions.copy() for lig in ligands]
            lig_initial_coms = [lig.center_of_mass() for lig in ligands]
            lig_rmsd: list[list[float]] = [[] for _ in ligands]
            lig_com_drift: list[list[float]] = [[] for _ in ligands]

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
                    lig_com_drift[i].append(
                        # distance between start and current ligand position
                        # ignores PBC, but we've already centered the traj
                        mda.lib.distances.calc_bonds(lig.center_of_mass(), lig_initial_coms[i])
                    )

            protein_2d = twoD_RMSD(prot_positions, w=None) if prot else None
            protein_rmsd_out = prot_rmsd if prot else None

            ligands_data = None
            if ligands:
                single_ligands = [
                    SingleLigandRMSData(
                        rmsd=lig_rmsd[i],
                        com_drift=lig_com_drift[i],
                        resname=lig.residues[0].resname,
                        resid=lig.residues[0].resid,
                        segid=lig.residues[0].segid,
                    )
                    for i, lig in enumerate(ligands)
                ]
                ligands_data = LigandsRMSData(single_ligands)

            states.append(
                StateRMSData(
                    protein_rmsd=protein_rmsd_out,
                    protein_2d_rmsd=protein_2d,
                    ligands=ligands_data,
                ),
            )

            time = list(np.arange(len(u.trajectory))[::skip] * u.trajectory.dt)

    return RMSResults(time_ps=time, states=states)


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
