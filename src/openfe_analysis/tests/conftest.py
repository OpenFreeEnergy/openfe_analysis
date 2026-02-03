import pathlib
from importlib import resources

import pooch
import pytest

ZENODO_DOI = "doi:10.5281/zenodo.18378051"

ZENODO_FILES = {
    "openfe_analysis_simulation_output.tar.gz": "md5:7f0babaac3dc8f7dd2db63cb79dff00f",
    "openfe_analysis_skipped.tar.gz": "md5:ac42219bde9da3641375adf3a9ddffbf",
}

POOCH_CACHE = pathlib.Path(pooch.os_cache("openfe_analysis"))
POOCH_CACHE.mkdir(parents=True, exist_ok=True)

ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url=ZENODO_DOI,
    registry=ZENODO_FILES,
)


def _fetch_and_untar_once(filename: str) -> pathlib.Path:
    # If already untarred, reuse it
    untar_dir = POOCH_CACHE / f"{filename}.untar"
    if untar_dir.exists():
        return untar_dir

    # Otherwise fetch + untar
    paths = ZENODO_RBFE_DATA.fetch(filename, processor=pooch.Untar())

    return pathlib.Path(paths[0]).parent


@pytest.fixture(scope="session")
def rbfe_output_data_dir() -> pathlib.Path:
    untar_dir = _fetch_and_untar_once("openfe_analysis_simulation_output.tar.gz")
    return untar_dir / "openfe_analysis_simulation_output"



@pytest.fixture(scope="session")
def rbfe_skipped_data_dir() -> pathlib.Path:
    untar_dir = _fetch_and_untar_once("openfe_analysis_skipped.tar.gz")
    return untar_dir / "openfe_analysis_skipped"



@pytest.fixture(scope="session")
def simulation_nc(rbfe_output_data_dir) -> pathlib.Path:
    return rbfe_output_data_dir / "simulation.nc"


@pytest.fixture(scope="session")
def simulation_skipped_nc(rbfe_skipped_data_dir) -> pathlib.Path:
    return rbfe_skipped_data_dir / "simulation.nc"


@pytest.fixture(scope="session")
def hybrid_system_pdb(rbfe_output_data_dir) -> pathlib.Path:
    return rbfe_output_data_dir / "hybrid_system.pdb"


@pytest.fixture(scope="session")
def hybrid_system_skipped_pdb(rbfe_skipped_data_dir) -> pathlib.Path:
    return rbfe_skipped_data_dir / "hybrid_system.pdb"


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
