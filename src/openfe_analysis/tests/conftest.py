from importlib import resources

import pathlib
import pooch
import pytest
import tempfile
import shutil

POOCH_CACHE = pathlib.Path(pooch.os_cache("openfe_analysis"))
ZENODO_RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="doi:10.5281/zenodo.17916322",
    registry={
       "openfe_analysis_simulation_output.tar.gz":"md5:09752f2c4e5b7744d8afdee66dbd1414",
       "openfe_analysis_skipped.tar.gz": "md5:3840d044299caacc4ccd50e6b22c0880",
    },
)


def _fetch_and_untar(archive_name: str, extracted_name: str) -> pathlib.Path:
    archive = ZENODO_RBFE_DATA.fetch(archive_name)

    final_dir = (
        POOCH_CACHE
        / f"{archive_name}.untar"
        / extracted_name
    )

    # Fast path: already extracted
    if final_dir.exists():
        return final_dir

    tmp_root = tempfile.mkdtemp(dir=POOCH_CACHE)
    try:
        pooch.Untar(extract_dir=tmp_root)(
            archive,
            action="fetch",
            pooch=ZENODO_RBFE_DATA,
        )

        extracted_dir = pathlib.Path(tmp_root) / extracted_name
        final_dir.parent.mkdir(parents=True, exist_ok=True)

        # Atomic on POSIX filesystems
        shutil.move(extracted_dir, final_dir)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return final_dir


@pytest.fixture(scope="session")
def rbfe_output_data_dir() -> pathlib.Path:
    return _fetch_and_untar(
        "openfe_analysis_simulation_output.tar.gz",
        "openfe_analysis_simulation_output",
    )


@pytest.fixture(scope="session")
def rbfe_skipped_data_dir() -> pathlib.Path:
    return _fetch_and_untar(
        "openfe_analysis_skipped.tar.gz",
        "openfe_analysis_skipped",
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
