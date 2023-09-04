from MDAnalysis.coordinates.base import ReaderBase, Timestep
import netCDF4 as nc
from openfe.utils import handle_trajectories
from openff.units import unit
from typing import Optional


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

    format = 'openfe RFE'

    def __init__(self, filename, convert_units=True, **kwargs):
        """
        Parameters
        ----------
        filename : pathlike
          path to the .nc file
        convert_units : bool
          convert positions to A
        """
        self._state_id = kwargs.pop('state_id', None)
        self._replica_id = kwargs.pop('replica_id', None)
        if not ((self._state_id is None) ^ (self._replica_id is None)):
            raise ValueError("Specify one and only one of state or replica, "
                             f"got state id={self._state_id} "
                             f"replica_id={self._replica_id}")

        super().__init__(filename, convert_units, **kwargs)

        self._dataset = nc.Dataset(filename)
        self._n_atoms = self._dataset.dimensions['atom'].size
        self.ts = Timestep(self._n_atoms)
        self._read_frame(0)

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

        return self.ts

    def _reopen(self):
        self._frame_index = -1

    def close(self):
        self._dataset.close()
