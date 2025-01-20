from MDAnalysis.coordinates.base import ReaderBase, Timestep
import netCDF4 as nc
from openff.units import unit
import numpy as np
import yaml
from typing import Optional


from openfe_analysis.utils import multistate, serialization
from openfe_analysis.utils.multistate import _determine_position_indices


def _determine_iteration_dt(dataset) -> float:
    """
    Find out the timestep between each frame in the trajectory.

    Parameters
    ----------
    dataset : nc.Dataset
      Dataset holding the multistatereporter generated NetCDF file.

    Returns
    -------
    float
      The timestep in units of picoseconds.

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
        dataset.groups['mcmc_moves']['move0'][0],
        Loader=serialization.UnitedYamlLoader,
    )

    try:
        dt = mcmc_move_data['n_steps'] * mcmc_move_data['timestep']
    except KeyError:
        msg = "Either `n_steps` or `timestep` are missing from the MCMC move"
        raise KeyError(msg)

    return dt.to('picosecond').m


class FEReader(ReaderBase):
    """A MDAnalysis Reader for NetCDF files created by
    `openmmtools.multistate.MultistateReporter`

    Looks along a multistate NetCDF file along one of two axes:
      - constant state/lambda (varying replica)
      - constant replica (varying lambda)
    """
    _state_id: Optional[int]
    _replica_id: Optional[int]
    _frame_index: int
    _dataset: nc.Dataset
    _dataset_owner: bool

    format = 'MultistateReporter'

    units = {
        'time': 'ps',
        'length': 'nanometer'
    }

    def __init__(
        self, filename, convert_units=True,
        state_id=None, replica_id=None, **kwargs
    ):
        """
        Parameters
        ----------
        filename : pathlike or nc.Dataset
          path to the .nc file
        convert_units : bool
          convert positions to Angstrom
        state_id : Optional[int]
          The hamiltonian state index to extract. Must be defined if
          ``replica_id`` is not defined.
        replica_id : Optional[int]
          The replica index to extract. Must be defined if ``state_id``
          is not defined.
        """
        if not ((state_id is None) ^ (replica_id is None)):
            raise ValueError("Specify one and only one of state or replica, "
                             f"got state id={state_id} "
                             f"replica_id={replica_id}")

        super().__init__(filename, convert_units, **kwargs)

        if isinstance(filename, nc.Dataset):
            self._dataset = filename
            self._dataset_owner = False
        else:
            self._dataset = nc.Dataset(filename)
            self._dataset_owner = True

        # Handle the negative ID case
        if state_id is not None and state_id < 0:
            state_id = range(self._dataset.dimensions['state'].size)[state_id]

        if replica_id is not None and replica_id < 0:
            replica_id = range(self._dataset.dimensions['replica'].size)[replica_id]

        self._state_id = state_id
        self._replica_id = replica_id

        self._n_atoms = self._dataset.dimensions['atom'].size
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
        with nc.Dataset(filename) as ds:
            n_atoms = ds.dimensions['atom'].size
        return n_atoms

    def _read_next_timestep(self, ts=None) -> Timestep:
        if (self._frame_index + 1) >= len(self):
            raise EOFError
        return self._read_frame(self._frame_index + 1)

    def _read_frame(self, frame: int) -> Timestep:
        self._frame_index = frame

        if self._state_id is not None:
            rep = multistate._state_to_replica(
                self._dataset,
                self._state_id,
                self._frames[self._frame_index]
            )
        else:
            rep = self._replica_id

        pos = multistate._replica_positions_at_frame(
            self._dataset,
            rep,
            self._frames[self._frame_index]
        )
        dim = multistate._get_unitcell(
            self._dataset,
            rep,
            self._frames[self._frame_index]
        )

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
        return self._dt

    def _reopen(self):
        self._frame_index = -1

    def close(self):
        if self._dataset_owner:
            self._dataset.close()
