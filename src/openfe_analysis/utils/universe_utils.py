from pathlib import Path

import MDAnalysis as mda
import netCDF4 as nc

from ..reader import FEReader


def create_universe_single_state(
    top: Path | mda.core.topology.Topology, trj: nc.Dataset, state: int
) -> mda.Universe:
    """
    Construct a raw MDAnalysis Universe for a single thermodynamic state.

    Parameters
    ----------
    top : pathlib.Path | mda.core.topology.Topology
        Path to a topology file (e.g. PDB).
    trj : nc.Dataset
        Open NetCDF dataset produced by
        ``openmmtools.multistate.MultiStateReporter``.
    state : int
        Thermodynamic state index to extract from the multistate trajectory.

    Returns
    -------
    mda.Universe
        A Universe with no trajectory transformations applied.
    """
    return mda.Universe(
        top,
        trj,
        index=state,
        index_method="state",
        format=FEReader,
    )
