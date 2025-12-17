from importlib import resources

import pathlib
import pooch
import pytest
import urllib.request


try:
    urllib.request.urlopen('https://www.google.com')
except:  # -no-cov-
    HAS_INTERNET = False
else:
    HAS_INTERNET = True


POOCH_CACHE = pooch.os_cache("openfe_analysis")
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17916322",
    registry={
       "openfe_analysis_simulation_output.tar.gz":"md5:09752f2c4e5b7744d8afdee66dbd1414",
       "openfe_analysis_skipped.tar.gz": "md5:3840d044299caacc4ccd50e6b22c0880",
    },
    retry_if_failed=5,
)

@pytest.fixture(scope="session")
def rbfe_output_data_dir() -> pathlib.Path:
    ZENODO_RBFE_DATA.fetch("openfe_analysis_simulation_output.tar.gz", processor=pooch.Untar())
    result_dir = pathlib.Path(POOCH_CACHE) / "openfe_analysis_simulation_output.tar.gz.untar/openfe_analysis_simulation_output/"
    return result_dir

@pytest.fixture(scope="session")
def rbfe_skipped_data_dir() -> pathlib.Path:
    ZENODO_RBFE_DATA.fetch("openfe_analysis_skipped.tar.gz", processor=pooch.Untar())
    result_dir = pathlib.Path(POOCH_CACHE) / "openfe_analysis_skipped.tar.gz.untar/openfe_analysis_skipped/"
    return result_dir

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


@pytest.fixture(scope='session')
def mcmc_serialized():
    return (
        "_serialized__class_name: LangevinDynamicsMove\n"
        "_serialized__module_name: openmmtools.mcmc\n"
        "collision_rate: !Quantity\n  unit: /picosecond\n  value: 1\n"
        "constraint_tolerance: 1.0e-06\nn_restart_attempts: 20\n"
        "n_steps: 625\nreassign_velocities: false\n"
        "timestep: !Quantity\n  unit: femtosecond\n  value: 4\n"
    )
