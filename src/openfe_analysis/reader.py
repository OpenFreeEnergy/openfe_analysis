import pathlib
from typing import Literal, Optional

import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import yaml
from MDAnalysis.coordinates.base import ReaderBase, Timestep
from openff.units import unit

from openfe_analysis.utils import multistate, serialization
from openfe_analysis.utils.multistate import _determine_position_indices


def _create_universe_single_state(top, trj, state):
    return mda.Universe(
        top,
        trj,
        index=state,
        index_method="state",
        format=FEReader,
    )


def _determine_iteration_dt(dataset) -> float:
    """
    Determine the time increment between successive iterations
    in a MultiStateReporter trajectory.

    Parameters
    ----------
    dataset : nc.Dataset
      NetCDF dataset produced by ``openmmtools.multistate.MultiStateReporter``.

    Returns
    -------
    float
      The time between successive iterations, in picoseconds.

    Raises
    ------
    KeyError
      If either `timestep` or `n_steps` cannot be found in the
      zeroth MCMC move.

    Notes
    -----
    This assumes an MCMC move which serializes in a manner similar
    to `openmmtools.mcmc.LangevinDynamicsMove`, i.e. it must have
    both a `timestep` and `n_steps` defined, such that
        dt_iteration = n_steps * timestep
    """
    # Deserialize the MCMC move information for the 0th entry.
    mcmc_move_data = yaml.load(
        dataset.groups["mcmc_moves"]["move0"][0],
        Loader=serialization.UnitedYamlLoader,
    )

    try:
        dt = mcmc_move_data["n_steps"] * mcmc_move_data["timestep"]
    except KeyError:
        msg = "Either `n_steps` or `timestep` are missing from the MCMC move"
        raise KeyError(msg)

    return dt.to("picosecond").m


class FEReader(ReaderBase):
    """
    MDAnalysis Reader for NetCDF files created by
    `openmmtools.multistate.MultiStateReporter`

    Provides a 1D trajectory along either:

    - constant Hamiltonian state (`index_method="state"`)
    - constant replica (`index_method="replica"`)

    selected via the `index` argument.
    """

    _multistate_index: Optional[int]
    _index_method: Optional[str]
    _frame_index: int
    _dataset: nc.Dataset
    _dataset_owner: bool

    format = "MultiStateReporter"

    units = {"time": "ps", "length": "nanometer"}

    def __init__(
        self,
        filename: str | pathlib.Path | nc.Dataset,
        *,
        index: int,
        index_method: Literal["state", "replica"] = "state",
        convert_units: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        filename : pathlike or nc.Dataset
            Path to the .nc file or an open Dataset.
        index : int
            Index of the state or replica to extract. May be negative.
        index_method : {"state", "replica"}, default "state"
            Whether `index` refers to a Hamiltonian state or a replica.
        convert_units : bool
            Convert positions to Angstrom.
        """
        super().__init__(filename, convert_units, **kwargs)

        if isinstance(filename, nc.Dataset):
            self._dataset = filename
            self._dataset_owner = False
        else:
            self._dataset = nc.Dataset(filename)
            self._dataset_owner = True

        if index_method not in {"state", "replica"}:
            raise ValueError(f"index_method must be 'state' or 'replica', got {index_method}")

        self._index_method = index_method

        # Handle the negative ID case
        if index_method == "state":
            size = self._dataset.dimensions["state"].size
        else:
            size = self._dataset.dimensions["replica"].size

        self._multistate_index = index % size

        self._n_atoms = self._dataset.dimensions["atom"].size
        self.ts = Timestep(self._n_atoms)
        self._frames = _determine_position_indices(self._dataset)
        # The MDAnalysis trajectory "dt" is the iteration dt
        # multiplied by the number of iterations between frames.
        self._dt = _determine_iteration_dt(self._dataset) * np.diff(self._frames)[0]
        self._read_frame(0)

    @staticmethod
    def _format_hint(thing) -> bool:
        # can pass raw nc datasets through to reduce open/close operations
        return isinstance(thing, nc.Dataset)

    @property
    def multistate_index(self) -> int:
        return self._multistate_index

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    @property
    def index_method(self) -> str:
        return self._index_method

    @staticmethod
    def parse_n_atoms(filename, **kwargs) -> int:
        """
        Determine the number of atoms stored in a MultiStateReporter NetCDF file.

        Parameters
        ----------
        filename : path-like
            Path to the NetCDF file.

        Returns
        -------
        int
            Number of atoms in the system.
        """
        with nc.Dataset(filename) as ds:
            n_atoms = ds.dimensions["atom"].size
        return n_atoms

    def _read_next_timestep(self, ts=None) -> Timestep:
        # Advance the trajectory by one frame.
        if (self._frame_index + 1) >= len(self):
            raise EOFError
        return self._read_frame(self._frame_index + 1)

    def _read_frame(self, frame: int) -> Timestep:
        # Read a single trajectory frame.
        self._frame_index = frame

        frame = self._frames[self._frame_index]

        if self._index_method == "state":
            rep = multistate._state_to_replica(
                self._dataset,
                self._multistate_index,
                frame,
            )
        else:
            rep = self._multistate_index

        pos = multistate._replica_positions_at_frame(self._dataset, rep, frame)
        dim = multistate._get_unitcell(self._dataset, rep, frame)

        if pos is None:
            errmsg = (
                "NetCDF dataset frame without positions was accessed "
                "this likely indicates that the reader failed to work out "
                "the write frequency and there is a deeper issue with how "
                "this file was written."
            )
            raise RuntimeError(errmsg)

        # Convert to base MDAnalysis distance units (Angstrom) if requested
        if self.convert_units:
            self.ts.positions = (pos.to(unit.angstrom)).m
        else:
            self.ts.positions = pos.m
        self.ts.dimensions = dim
        self.ts.frame = self._frame_index
        self.ts.time = self._frame_index * self._dt

        return self.ts

    @property
    def dt(self) -> float:
        # Time difference between successive trajectory frames.
        return self._dt

    def _reopen(self):
        self._frame_index = -1

    def close(self):
        if self._dataset is not None:
            if self._dataset_owner:
                self._dataset.close()
            self._dataset = None
