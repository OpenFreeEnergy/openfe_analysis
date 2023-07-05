from MDAnalysis.coordinates.base import ReaderBase, Timestep
import netCDF4 as nc
from openfe.utils import handle_trajectories
from openff.units import unit


class FEReader(ReaderBase):
    """A MDAnalysis Reader for nc files created by openfe RFE Protocol"""
    _state_id: int
    _frame_index: int
    _dataset

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
        try:
            self._state_id = kwargs.pop('state_id')
        except KeyError:
            raise ValueError("Specify the state_id")

        super().__init__(filename, convert_units, kwargs)

        self._dataset = nc.Dataset(filename)
        self._n_atoms = ds.dimensions['atom'].size
        self._read_frame(0)

    @property
    def n_frames(self) -> int:
        return self._dataset.dimensions['iteration'].size

    @staticmethod
    def parse_n_atoms(filename, **kwargs) -> int:
        with nc.Dataset(filename) as ds:
            n_atoms = ds.dimensions['atom'].size

        return n_atoms

    def _read_next_timestep(self, ts=None) -> Timestep:
        return self._read_frame(self._frame_index + 1)

    def _read_frame(self, frame: int) -> Timestep:
        self._frame_index = frame

        ts = Timestep(self._n_atoms)

        pos = handle_trajectories._state_positions_at_frame(
            self._dataset,
            self._state_id,
            self._frame_index)
        dim = handle_trajectories._get_unitcell(
            self._dataset,
            self._state_id,
            self._frame_index)

        ts.positions = (pos.to(unit.angstrom)).m
        ts.dimensions = dim
        ts.frame = self._frame_index

        return ts

    def _reopen(self):
        self._frame_index = -1

    def close(self):
        self._dataset.close()
