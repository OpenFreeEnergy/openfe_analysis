from typing import Optional

import netCDF4 as nc
import numpy as np
import yaml
from MDAnalysis.coordinates.base import ReaderBase, Timestep
from openff.units import unit

from openfe_analysis.utils import multistate, serialization
from openfe_analysis.utils.multistate import _determine_position_indices


def _determine_iteration_dt(dataset) -> float:
    """
    Determine the time increment between successive iterations
    in a MultiStateReporter trajectory.

    The timestep is inferred from the serialized MCMC move stored in the
    ``mcmc_moves`` group of the NetCDF file. Specifically, this assumes the
    move defines both a ``timestep`` and ``n_steps`` parameter, such that

        dt_iteration = n_steps * timestep

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
    both a `timestep` and `n_steps` defined.
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
    MDAnalysis trajectory reader for NetCDF files written by
    ``openmmtools.multistate.MultiStateReporter``.

    Looks along a multistate NetCDF file along one of two axes:
      - constant state/lambda (varying replica)
      - constant replica (varying lambda)
    Exactly one of ``state_id`` or ``replica_id`` must be specified.
    """

    _state_id: Optional[int]
    _replica_id: Optional[int]
    _frame_index: int
    _dataset: nc.Dataset
    _dataset_owner: bool

    format = "MultiStateReporter"

    units = {"time": "ps", "length": "nanometer"}

    def __init__(self, filename, convert_units=True, state_id=None, replica_id=None, **kwargs):
        """
        Parameters
        ----------
        filename : pathlike or nc.Dataset
          Path to a MultiStateReporter NetCDF file, or an already-open
          ``netCDF4.Dataset`` instance.
        convert_units : bool
          If ``True`` (default), positions are converted to Angstroms.
          Otherwise, raw OpenMM units (nanometers) are returned.
        state_id : Optional[int]
          The Hamiltonian state index to extract. Must be defined if
          ``replica_id`` is not defined. May be negative (see notes below).
        replica_id : Optional[int]
          The replica index to extract. Must be defined if ``state_id``
          is not defined. May be negative (see notes below).

        Raises
        ------
        ValueError
            If neither or both of ``state_id`` and ``replica_id`` are specified.

        Notes
        -----
        A negative index may be passed to either ``state_id`` or
        ``replica_id``. This will be interpreted as indexing in reverse
        starting from the last state/replica. For example, ``replica_id=-2``
        will select the before last replica.
        """
        if not ((state_id is None) ^ (replica_id is None)):
            raise ValueError(
                "Specify one and only one of state or replica, "
                f"got state id={state_id} "
                f"replica_id={replica_id}"
            )

        super().__init__(filename, convert_units, **kwargs)

        if isinstance(filename, nc.Dataset):
            self._dataset = filename
            self._dataset_owner = False
        else:
            self._dataset = nc.Dataset(filename)
            self._dataset_owner = True

        # Handle the negative ID case
        if state_id is not None and state_id < 0:
            state_id = range(self._dataset.dimensions["state"].size)[state_id]

        if replica_id is not None and replica_id < 0:
            replica_id = range(self._dataset.dimensions["replica"].size)[replica_id]

        self._state_id = state_id
        self._replica_id = replica_id

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
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def n_frames(self) -> int:
        return len(self._frames)

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

        if self._state_id is not None:
            rep = multistate._state_to_replica(
                self._dataset, self._state_id, self._frames[self._frame_index]
            )
        else:
            rep = self._replica_id

        pos = multistate._replica_positions_at_frame(
            self._dataset, rep, self._frames[self._frame_index]
        )
        dim = multistate._get_unitcell(self._dataset, rep, self._frames[self._frame_index])

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
        # Close the underlying NetCDF dataset if owned by this reader.
        if self._dataset_owner:
            self._dataset.close()
