import MDAnalysis as mda
from openfe_analysis import FEReader
import netCDF4 as nc
import pytest


def test_universe_creation(simulation_nc, hybrid_system_pdb):
    u = mda.Universe(hybrid_system_pdb, simulation_nc,
                     format='openfe rfe', state_id=0)

    assert u
    assert len(u.atoms) == 4782
    assert len(u.trajectory) == 501
    assert u.trajectory.dt == pytest.approx(1.0)


def test_universe_from_nc_file(simulation_nc, hybrid_system_pdb):
    ds = nc.Dataset(simulation_nc)

    u = mda.Universe(hybrid_system_pdb, ds,
                     format='openfe rfe', state_id=0)

    assert u
    assert len(u.atoms) == 4782


