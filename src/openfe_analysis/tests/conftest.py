import pathlib

import pooch
import pytest

ZENODO_DOI = "doi:10.5281/zenodo.17916321"

ZENODO_FILES = {
    "openfe_analysis_full.tar.gz": "md5:a51b1f8d98b91ab1a69a6f55508d07db",
    "openfe_analysis_skipped.tar.gz": "md5:ac42219bde9da3641375adf3a9ddffbf",
    "openfe_analysis_septop.tar.gz": "md5:4b47198c57025bd6e0c6cf76f864370a",
}

POOCH_CACHE = pathlib.Path(pooch.os_cache("openfe_analysis"))
POOCH_CACHE.mkdir(parents=True, exist_ok=True)

ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url=ZENODO_DOI,
    registry=ZENODO_FILES,
)


def _fetch_and_untar(dirname: str) -> pathlib.Path:
    ZENODO_RBFE_DATA.fetch(f"{dirname}.tar.gz", processor=pooch.Untar())
    cached_dir = pathlib.Path(f"{POOCH_CACHE}/{dirname}.tar.gz.untar/{dirname}")
    return cached_dir


@pytest.fixture(scope="session")
def rbfe_output_data_dir() -> pathlib.Path:
    cached_dir = _fetch_and_untar("openfe_analysis_full")
    return cached_dir


@pytest.fixture(scope="session")
def rbfe_skipped_data_dir() -> pathlib.Path:
    cached_dir = _fetch_and_untar("openfe_analysis_skipped")
    return cached_dir


@pytest.fixture(scope="session")
def rbfe_septop_data_dir() -> pathlib.Path:
    cached_dir = _fetch_and_untar("openfe_analysis_septop")
    return cached_dir


@pytest.fixture(scope="session")
def simulation_nc(rbfe_output_data_dir) -> pathlib.Path:
    return rbfe_output_data_dir / "simulation.nc"


@pytest.fixture(scope="session")
def simulation_skipped_nc(rbfe_skipped_data_dir) -> pathlib.Path:
    return rbfe_skipped_data_dir / "simulation.nc"


@pytest.fixture(scope="session")
def simulation_nc_septop(rbfe_septop_data_dir) -> pathlib.Path:
    return rbfe_septop_data_dir / "complex.nc"


@pytest.fixture(scope="session")
def system_septop(rbfe_septop_data_dir) -> pathlib.Path:
    return rbfe_septop_data_dir / "alchemical_system.pdb"


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
