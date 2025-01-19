import MDAnalysis as mda
from openfe_analysis.reader import FEReader, _determine_dt
import netCDF4 as nc
import pytest


def test_determine_dt(tmpdir, mcmc_serialized):
    with tmpdir.as_cwd():
        # create a fake dataset with a fake mcmc move group
        ds = nc.Dataset('foo', 'w', format='NETCDF3_64BIT_OFFSET')
        ds.groups['mcmc_moves'] = {
            'move0': [mcmc_serialized]
        }

        assert _determine_dt(ds) == 2.5


def test_determine_dt_keyerror(tmpdir, mcmc_serialized):
    with tmpdir.as_cwd():
        # create a fake dataset with fake mcmc move without timestep
        ds = nc.Dataset('foo', 'w', format='NETCDF3_64BIT_OFFSET')
        ds.groups['mcmc_moves'] = {
            'move0': [mcmc_serialized[:-51]]
        }

        with pytest.raises(KeyError, match="`n_steps` or `timestep` are"):
            _ = _determine_dt(ds)


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


