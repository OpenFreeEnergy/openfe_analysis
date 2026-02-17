import netCDF4 as nc
import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.units import unit

from openfe_analysis import __version__
from openfe_analysis.utils.multistate import (
    _create_new_dataset,
    _determine_position_indices,
    _get_unitcell,
    _replica_positions_at_frame,
    _state_to_replica,
    trajectory_from_multistate,
)


@pytest.fixture(scope="module")
def dataset(simulation_nc):
    ds = nc.Dataset(simulation_nc)
    yield ds
    ds.close()


@pytest.fixture(scope="module")
def skipped_dataset(simulation_skipped_nc):
    ds = nc.Dataset(simulation_skipped_nc)
    yield ds
    ds.close()


@pytest.mark.parametrize("state, frame, replica", [[0, 0, 0], [0, 1, 0], [0, -1, 2], [1, 100, 1]])
def test_state_to_replica(dataset, state, frame, replica):
    assert _state_to_replica(dataset, state, frame) == replica


def test_replica_positions_at_frame(dataset):
    pos = _replica_positions_at_frame(dataset, 1, -1)
    assert_allclose(
        pos[-3] * unit("nanometer"),
        np.array([4.674962, 2.110855, 0.844064]) * unit("nanometer"),
        atol=1e-6,
    )


def test_determine_position_indices_inconsistent(monkeypatch, dataset):
    # Force np.diff to return inconsistent spacing
    def fake_diff(x):
        return np.array([1, 2, 1])

    monkeypatch.setattr(np, "diff", fake_diff)

    with pytest.raises(ValueError, match="consistent frame rate"):
        _determine_position_indices(dataset)


def test_create_new_dataset(tmp_path):
    file_path = tmp_path / "foo.nc"
    with _create_new_dataset(file_path, 100, title="bar") as ds:
        # Test metadata
        assert ds.Conventions == "AMBER"
        assert ds.ConventionVersion == "1.0"
        assert ds.application == "openfe_analysis"
        assert ds.program == f"openfe_analysis {__version__}"
        assert ds.programVersion == f"{__version__}"
        assert ds.title == "bar"

        # Test dimensions
        assert ds.dimensions["frame"].size == 0
        assert ds.dimensions["spatial"].size == 3
        assert ds.dimensions["atom"].size == 100
        assert ds.dimensions["cell_spatial"].size == 3
        assert ds.dimensions["cell_angular"].size == 3
        assert ds.dimensions["label"].size == 5

        # Test variables
        assert ds.variables["coordinates"].units == "angstrom"
        assert ds.variables["coordinates"].get_dims()[0].name == "frame"
        assert ds.variables["coordinates"].get_dims()[1].name == "atom"
        assert ds.variables["coordinates"].get_dims()[2].name == "spatial"
        assert ds.variables["coordinates"].dtype.name == "float32"

        assert ds.variables["cell_lengths"].units == "angstrom"
        assert ds.variables["cell_lengths"].get_dims()[0].name == "frame"
        assert ds.variables["cell_lengths"].get_dims()[1].name == "cell_spatial"
        assert ds.variables["cell_lengths"].dtype.name == "float64"

        assert ds.variables["cell_angles"].units == "degree"
        assert ds.variables["cell_angles"].get_dims()[0].name == "frame"
        assert ds.variables["cell_angles"].get_dims()[1].name == "cell_angular"
        assert ds.variables["cell_angles"].dtype.name == "float64"


def test_get_unitcell(dataset):
    dims = _get_unitcell(dataset, 1, -1)
    assert_allclose(dims, [78.10947, 78.10947, 78.10947, 60.0, 60.0, 90.0])

    dims = _get_unitcell(dataset, 2, 1)
    assert_allclose(dims, [78.20665, 78.20665, 78.20665, 60.0, 60.0, 90.0])


def test_simulation_skipped_nc_no_positions_box_vectors_frame1(
    skipped_dataset,
):
    assert _get_unitcell(skipped_dataset, 1, 1) is None
    assert skipped_dataset.variables["positions"][1][0].mask.all()


def test_trajectory_invalid_index_method(tmp_path):
    dummy_input = tmp_path / "dummy.nc"
    dummy_output = tmp_path / "out.nc"

    # Create minimal NetCDF
    ds = nc.Dataset(dummy_input, "w", format="NETCDF3_64BIT_OFFSET")
    ds.createDimension("atom", 1)
    ds.createDimension("frame", 1)
    pos = ds.createVariable("positions", "f4", ("frame", "atom"))
    pos[:] = 0.0
    ds.close()

    with pytest.raises(ValueError, match="index_method must be 'state' or 'replica'"):
        trajectory_from_multistate(dummy_input, dummy_output, index=0, index_method="foo")


def test_trajectory_frame_without_positions(tmp_path):
    dummy_input = tmp_path / "dummy.nc"
    dummy_output = tmp_path / "out.nc"

    # Minimal NetCDF file
    with nc.Dataset(dummy_input, "w", format="NETCDF4") as ds:
        ds.createDimension("frame", 2)  # at least 2 frames
        ds.createDimension("replica", 1)
        ds.createDimension("atom", 1)
        ds.createDimension("spatial", 3)
        ds.createDimension("iteration", 2)  # at least 2 iterations

        positions = ds.createVariable("positions", "f4", ("frame", "replica", "atom", "spatial"))
        positions.units = "nanometer"
        positions[:] = np.ma.masked  # All positions masked

    # Expect RuntimeError due to missing positions
    with pytest.raises(RuntimeError, match="Frame without positions encountered"):
        trajectory_from_multistate(dummy_input, dummy_output, index=0, index_method="replica")


def test_trajectory_success(tmp_path):
    dummy_input = tmp_path / "dummy.nc"
    dummy_output = tmp_path / "out.nc"

    # Minimal valid NetCDF with positions, box vectors, and iteration dimension
    ds = nc.Dataset(dummy_input, "w", format="NETCDF3_64BIT_OFFSET")
    ds.createDimension("atom", 2)
    ds.createDimension("frame", 2)
    ds.createDimension("replica", 2)
    ds.createDimension("state", 2)
    ds.createDimension("spatial", 3)
    ds.createDimension("iteration", 2)  # Added for _determine_position_indices

    # positions: frame x replica x atom x spatial
    pos = ds.createVariable("positions", "f4", ("frame", "replica", "atom", "spatial"))
    pos.units = "nanometer"
    pos[:] = np.zeros((2, 2, 2, 3), dtype=np.float32)

    # box_vectors: frame x replica x 3 x 3
    bv = ds.createVariable("box_vectors", "f8", ("frame", "replica", "spatial", "spatial"))
    bv.units = "nanometer"
    bv[:] = np.tile(np.eye(3), (2, 2, 1, 1))

    # states: frame x replica
    st = ds.createVariable("states", "i4", ("frame", "replica"))
    st[:] = np.array([[0, 1], [0, 1]], dtype=np.int32)  # replica 0->state 0, replica1->state1

    ds.close()

    # Call function for replica extraction
    trajectory_from_multistate(dummy_input, dummy_output, index=1, index_method="replica")

    # Check output file exists and contains positions
    out_ds = nc.Dataset(dummy_output, "r")
    assert out_ds.variables["coordinates"].shape == (2, 2, 3)
    assert out_ds.variables["cell_lengths"].shape == (2, 3)
    assert out_ds.variables["cell_angles"].shape == (2, 3)
    out_ds.close()
