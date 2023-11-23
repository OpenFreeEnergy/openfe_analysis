from MDAnalysis.coordinates.base import ReaderBase, Timestep
import netCDF4 as nc
from openff.units import unit
from typing import Optional


from . import handle_trajectories


def _determine_dt(ds) -> float:
    # first grab integrator timestep
    mcmc_move_data = ds.groups['mcmc_moves']['move0'][0].split('\n')
    in_timestep = False
    for line in mcmc_move_data:
        if line.startswith('timestep'):
            in_timestep = True
        if in_timestep and line.strip().startswith('value'):
            timestep = float(line.split()[-1]) / 1000.  # convert to ps
            break
    else:
        raise ValueError("Didn't find timestep")
    # next get the save interval
    option_data = ds.variables['options'][0].split('\n')
    for line in option_data:
        if line.startswith('online_analysis_interval'):
            nsteps = float(line.split()[-1])
            break
    else:
        raise ValueError("Didn't find online_analysis_interval")

    return timestep * nsteps


class FEReader(ReaderBase):
    """A MDAnalysis Reader for nc files created by openfe RFE Protocol

    Looks along a multistate .nc file along one of two axes:
    - constant state/lambda (varying replica)
    - constant replica (varying lambda)
    """
    _state_id: Optional[int]
    _replica_id: Optional[int]
    _frame_index: int
    _dataset: nc.Dataset
    _dataset_owner: bool

    format = 'openfe RFE'

    def __init__(self, filename, convert_units=True, **kwargs):
        """
        Parameters
        ----------
        filename : pathlike or nc.Dataset
          path to the .nc file
        convert_units : bool
          convert positions to A
        """
        s_id = kwargs.pop('state_id', None)
        r_id = kwargs.pop('replica_id', None)
        if not ((s_id is None) ^ (r_id is None)):
            raise ValueError("Specify one and only one of state or replica, "
                             f"got state id={s_id} "
                             f"replica_id={r_id}")

        super().__init__(filename, convert_units, **kwargs)

        if isinstance(filename, nc.Dataset):
            self._dataset = filename
            self._dataset_owner = False
        else:
            self._dataset = nc.Dataset(filename)
            self._dataset_owner = True

        # if we have a negative indexed state_id or replica_id, convert this
        if s_id is not None and s_id < 0:
            s_id = range(self._dataset.dimensions['state'].size)[s_id]
        elif r_id is not None and r_id < 0:
            r_id = range(self._dataset.dimensions['replica'].size)[r_id]
        self._state_id = s_id
        self._replica_id = r_id

        self._n_atoms = self._dataset.dimensions['atom'].size
        self.ts = Timestep(self._n_atoms)
        self._dt = _determine_dt(self._dataset)
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
        return self._dataset.dimensions['iteration'].size

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
            rep = handle_trajectories._state_to_replica(
                self._dataset,
                self._state_id,
                self._frame_index
            )
        else:
            rep = self._replica_id

        pos = handle_trajectories._replica_positions_at_frame(
            self._dataset,
            rep,
            self._frame_index)
        dim = handle_trajectories._get_unitcell(
            self._dataset,
            rep,
            self._frame_index)

        self.ts.positions = (pos.to(unit.angstrom)).m
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
