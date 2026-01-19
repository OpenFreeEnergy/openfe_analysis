import netCDF4 as nc
import numpy as np
import pytest
from numpy.testing import assert_allclose
from openff.units import unit

from openfe_analysis import __version__
from openfe_analysis.utils.multistate import (
    _create_new_dataset,
    _get_unitcell,
    _replica_positions_at_frame,
    _state_to_replica,
)


@pytest.fixture(scope="module")
def dataset(simulation_nc):
    ds = nc.Dataset(simulation_nc)
    yield ds
    ds.close()

@pytest.fixture()
def skipped_dataset(simulation_skipped_nc):
    ds = nc.Dataset(simulation_skipped_nc)
    yield ds
    ds.close()


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("state, frame, replica", [[0, 0, 0], [0, 1, 3], [0, -1, 7], [3, 100, 6]])
def test_state_to_replica(dataset, state, frame, replica):
    assert _state_to_replica(dataset, state, frame) == replica


@pytest.mark.flaky(reruns=3)
def test_replica_positions_at_frame(dataset):
    pos = _replica_positions_at_frame(dataset, 1, -1)
    assert_allclose(
        pos[-3] * unit("nanometer"), np.array([0.6037003, 7.2835016, 5.804355]) * unit("nanometer")
    )


def test_create_new_dataset(tmpdir):
    with tmpdir.as_cwd():
        ds = _create_new_dataset("foo.nc", 100, title="bar")

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

        ds.close()


def test_get_unitcell(dataset):
    dims = _get_unitcell(dataset, 7, -1)
    assert_allclose(dims, [82.12723, 82.12723, 82.12723, 90.0, 90.0, 90.0])

    dims = _get_unitcell(dataset, 3, 1)
    assert_allclose(dims, [82.191055, 82.191055, 82.191055, 90.0, 90.0, 90.0])


def test_simulation_skipped_nc_no_positions_box_vectors_frame1(
    skipped_dataset,
):

    assert _get_unitcell(skipped_dataset, 1, 1) is None
    assert skipped_dataset.variables["positions"][1][0].mask.all()
