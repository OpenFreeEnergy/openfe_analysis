from importlib import resources

import pathlib
import pooch
import pytest
from filelock import FileLock

POOCH_CACHE = pooch.os_cache("openfe_analysis")
POOCH_CACHE.mkdir(parents=True, exist_ok=True)
LOCKFILE = POOCH_CACHE / "prepare.lock"
READY_FLAG = POOCH_CACHE / "data_ready.flag"

ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17916322",
    registry={
       "openfe_analysis_simulation_output.tar.gz":"md5:09752f2c4e5b7744d8afdee66dbd1414",
       "openfe_analysis_skipped.tar.gz": "md5:3840d044299caacc4ccd50e6b22c0880",
    },
)

def _prepare_data():
    """Download and extract large test data once per machine."""
    if READY_FLAG.exists():
        return

    with FileLock(str(LOCKFILE)):
        if READY_FLAG.exists():
            return

        ZENODO_RBFE_DATA.fetch(
            "openfe_analysis_simulation_output.tar.gz",
            processor=pooch.Untar(),
        )
        ZENODO_RBFE_DATA.fetch(
            "openfe_analysis_skipped.tar.gz",
            processor=pooch.Untar(),
        )

        READY_FLAG.touch()


_prepare_data()

@pytest.fixture(scope="session")
def rbfe_output_data_dir() -> pathlib.Path:
    return (
        POOCH_CACHE
        / "openfe_analysis_simulation_output.tar.gz.untar"
        / "openfe_analysis_simulation_output"
    )


@pytest.fixture(scope="session")
def rbfe_skipped_data_dir() -> pathlib.Path:
    return (
        POOCH_CACHE
        / "openfe_analysis_skipped.tar.gz.untar"
        / "openfe_analysis_skipped"
    )

@pytest.fixture(scope="session")
def simulation_nc(rbfe_output_data_dir) -> pathlib.Path:
    return rbfe_output_data_dir/"simulation.nc"


@pytest.fixture(scope="session")
def simulation_skipped_nc(rbfe_skipped_data_dir) -> pathlib.Path:
    return rbfe_skipped_data_dir/"simulation.nc"


@pytest.fixture(scope="session")
def hybrid_system_pdb(rbfe_output_data_dir) -> pathlib.Path:
    return rbfe_output_data_dir/"hybrid_system.pdb"


@pytest.fixture(scope="session")
def hybrid_system_skipped_pdb(rbfe_skipped_data_dir)->pathlib.Path:
    return rbfe_skipped_data_dir/"hybrid_system.pdb"


@pytest.fixture(scope="session")
def mcmc_serialized():
    return (
        "_serialized__class_name: LangevinDynamicsMove\n"
        "_serialized__module_name: openmmtools.mcmc\n"
        "collision_rate: !Quantity\n  unit: /picosecond\n  value: 1\n"
        "constraint_tolerance: 1.0e-06\nn_restart_attempts: 20\n"
        "n_steps: 625\nreassign_velocities: false\n"
        "timestep: !Quantity\n  unit: femtosecond\n  value: 4\n"
    )
