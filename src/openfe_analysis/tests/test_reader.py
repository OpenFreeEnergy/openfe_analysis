import MDAnalysis as mda
import netCDF4 as nc
import numpy as np
import pytest
from numpy.testing import assert_allclose

from openfe_analysis.reader import FEReader, _determine_iteration_dt, _determine_position_indices


def test_determine_dt(tmp_path, mcmc_serialized):
    file_path = tmp_path / "foo.nc"
    # create a fake dataset with a fake mcmc move group
    with nc.Dataset(file_path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.groups["mcmc_moves"] = {"move0": [mcmc_serialized]}

        assert _determine_iteration_dt(ds) == 2.5


def test_determine_dt_keyerror(tmp_path, mcmc_serialized):
    file_path = tmp_path / "foo.nc"
    # create a fake dataset with fake mcmc move without timestep
    with nc.Dataset(file_path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.groups["mcmc_moves"] = {"move0": [mcmc_serialized[:-51]]}

        with pytest.raises(KeyError, match="`n_steps` or `timestep` are"):
            _ = _determine_iteration_dt(ds)


def test_determine_position_indices_warns_for_old_nc(tmp_path):
    ncfile = tmp_path / "old.nc"

    # Create a minimal NetCDF file with an iteration dimension
    with nc.Dataset(ncfile, "w") as ds:
        ds.createDimension("iteration", 10)
        # IMPORTANT: do NOT set PositionInterval

    # Reopen for reading and test
    with nc.Dataset(ncfile) as ds:
        with pytest.warns(UserWarning, match="This is an older NetCDF file that"):
            indices = _determine_position_indices(ds)

    # Sanity check: should default to every iteration
    assert np.array_equal(indices, np.arange(10))


def test_universe_creation(simulation_nc, hybrid_system_pdb):
    u = mda.Universe(hybrid_system_pdb, simulation_nc, format=FEReader, state_id=0)

    # Check that a Universe exists
    assert u
    # Test the basics
    assert len(u.atoms) == 4763
    assert len(u.trajectory) == 501
    assert u.trajectory.dt == pytest.approx(1.0)
    for inx, ts in enumerate(u.trajectory):
        assert ts.time == inx
    assert u.trajectory.totaltime == pytest.approx(500)

    # Check the dimensions & positions of the first frame
    # Note: we multipy the positions by 10 since it's stored in nm
    assert_allclose(
        u.atoms[:3].positions,
        np.array(
            [
                [3.0412218, 5.1911503, 0.5535536],
                [3.0970716, 5.1305336, 0.462929],
                [2.9869991, 5.1086277, 0.6692577],
            ]
        )
        * 10,
    )
    assert_allclose(u.dimensions, [78.11549, 78.11549, 78.11549, 60, 60, 90])

    # Now check the second frame
    u.trajectory[1]
    assert u.trajectory.time == pytest.approx(1.0)
    assert_allclose(
        u.atoms[4:7].positions,
        np.array(
            [
                [2.9334116, 5.0555897, 0.6874318],
                [3.0383892, 4.9295105, 0.6182875],
                [3.0504719, 5.2566086, 0.5931664],
            ]
        )
        * 10,
        atol=1e-6,
    )
    assert_allclose(u.dimensions, [78.141495, 78.141495, 78.141495, 60.0, 60.0, 90.0])
    u.trajectory.close()


def test_universe_from_nc_file(simulation_skipped_nc, hybrid_system_skipped_pdb):
    with nc.Dataset(simulation_skipped_nc) as ds:
        u = mda.Universe(hybrid_system_skipped_pdb, ds, format="MultiStateReporter", state_id=0)

        assert u
        assert len(u.atoms) == 9178
        assert len(u.trajectory) == 51
        assert u.trajectory.dt == pytest.approx(100.0)


def test_universe_creation_noconversion(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = mda.Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        format=FEReader,
        state_id=0,
        convert_units=False,
    )
    assert u.trajectory.ts.frame == 0
    assert_allclose(
        u.atoms[:3].positions,
        np.array(
            [
                [7.958488, 2.319872, -0.927927],
                [7.976206, 2.407798, -0.810073],
                [7.981613, 2.526843, -0.828505],
            ]
        ),
        atol=1e-6,
    )
    u.trajectory.close()


def test_fereader_negative_state(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = mda.Universe(hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, state_id=-1)

    assert u.trajectory._state_id == 10
    assert u.trajectory._replica_id is None
    u.trajectory.close()


def test_fereader_negative_replica(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = mda.Universe(
        hybrid_system_skipped_pdb, simulation_skipped_nc, format=FEReader, replica_id=-2
    )

    assert u.trajectory._state_id is None
    assert u.trajectory._replica_id == 9
    u.trajectory.close()


@pytest.mark.parametrize("rep_id, state_id", [[None, None], [1, 1]])
def test_fereader_replica_state_id_error(
    simulation_skipped_nc, hybrid_system_skipped_pdb, rep_id, state_id
):
    with pytest.raises(ValueError, match="Specify one and only one"):
        _ = mda.Universe(
            hybrid_system_skipped_pdb,
            simulation_skipped_nc,
            format=FEReader,
            state_id=state_id,
            replica_id=rep_id,
        )


def test_simulation_skipped_nc(simulation_skipped_nc, hybrid_system_skipped_pdb):
    u = mda.Universe(
        hybrid_system_skipped_pdb,
        simulation_skipped_nc,
        format=FEReader,
        replica_id=0,
    )
    assert len(u.trajectory) == 51
    assert u.trajectory.n_frames == 51
    assert u.trajectory.dt == 100
    times = np.arange(0, 5001, 100)
    for inx, ts in enumerate(u.trajectory):
        assert ts.time == times[inx]
        # Positions are not all zero since PBC is not removed
        assert np.any(u.atoms.positions != 0)
    with pytest.raises(mda.exceptions.NoDataError, match="This Timestep has no velocities"):
        u.atoms.velocities
    u.trajectory.close()
