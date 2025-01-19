import MDAnalysis as mda
from openfe_analysis.reader import FEReader, _determine_dt
import netCDF4 as nc
from numpy.testing import assert_allclose
import numpy as np
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
                     format=FEReader, state_id=0)

    # Check that a Universe exists
    assert u

    # Test the basics
    assert len(u.atoms) == 4782
    assert len(u.trajectory) == 501
    assert u.trajectory.dt == pytest.approx(1.0)
    assert u.trajectory.time == pytest.approx(0.0)
    assert u.trajectory.totaltime == pytest.approx(500)

    # Check the dimensions & positions of the first frame
    # Note: we multipy the positions by 10 since it's stored in nm
    assert_allclose(
        u.atoms[:3].positions,
        np.array([[6.51474, -1.7640617, 8.406607],
                  [6.641961, -1.8410535, 8.433087],
                  [6.71369, -1.8112476, 8.533738]]) * 10,
    )
    assert_allclose(
        u.dimensions,
        [82.06851, 82.06851, 82.06851, 90., 90., 90.]
    )

    # Now check the second frame
    u.trajectory[1]
    assert u.trajectory.time == pytest.approx(1.0)
    assert_allclose(
        u.atoms[4:7].positions,
        np.array([[6.78754, -1.2783755, 8.433636],
                  [6.62524, -1.333609, 8.399696],
                  [6.744502, -1.5663723, 8.332421]]) * 10,
    )
    assert_allclose(
        u.dimensions,
        [82.191055, 82.191055, 82.191055, 90., 90., 90.]
    )

    # Now check the last frame
    u.trajectory[-1]
    assert u.trajectory.time == pytest.approx(500.0)
    assert_allclose(
        u.atoms[-3:].positions,
        np.array(
            [[2.9948092, 7.7675443, 0.19704354],
             [0.95652354, 2.99566, 1.3466661],
             [4.0027137, 4.695961, 3.6892936]]
        ) * 10,
    )
    assert_allclose(
        u.dimensions,
        [82.12723, 82.12723, 82.12723, 90., 90., 90.]
    )

    # Finally we rewind to the second frame to make sure that's possible
    u.trajectory[1]
    assert u.trajectory.time == pytest.approx(1.0)
    assert_allclose(
        u.atoms[4:7].positions,
        np.array([[6.78754, -1.2783755, 8.433636],
                  [6.62524, -1.333609, 8.399696],
                  [6.744502, -1.5663723, 8.332421]]) * 10,
    )
    assert_allclose(
        u.dimensions,
        [82.191055, 82.191055, 82.191055, 90., 90., 90.]
    )


def test_universe_from_nc_file(simulation_nc, hybrid_system_pdb):
    ds = nc.Dataset(simulation_nc)

    u = mda.Universe(hybrid_system_pdb, ds,
                     format='MultiStateReporter', state_id=0)

    assert u
    assert len(u.atoms) == 4782
    assert len(u.trajectory) == 501
    assert u.trajectory.dt == pytest.approx(1.0)


def test_universe_creation_noconversion(simulation_nc, hybrid_system_pdb):
    u = mda.Universe(hybrid_system_pdb, simulation_nc,
                     format=FEReader, state_id=0, convert_units=False)

    assert_allclose(
        u.atoms[:3].positions,
        np.array([[6.51474, -1.7640617, 8.406607],
                  [6.641961, -1.8410535, 8.433087],
                  [6.71369, -1.8112476, 8.533738]]),
    )

