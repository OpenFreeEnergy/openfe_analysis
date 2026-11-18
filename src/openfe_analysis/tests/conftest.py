import pathlib
from importlib import resources
import fcntl

import numpy as np
import pooch
import pytest
from rdkit import Chem

DATA_FILES = {
    "openfe_analysis_full.tar.gz": "sha256:cd54a67a12318249de327bb46df3074ba028e98734543c6d335135e3e2695f24",
    "openfe_analysis_skipped.tar.gz": "sha256:b19ef80a55e35a8442143cb4cf6b3fa80c6bbf84c5a7547f359360e2690af984",
    "openfe_analysis_septop.tar.gz": "sha256:366efd207ba56045ef4f814989de20debe70e11a182a0d4d1fabaaa171f89cdc",
}

POOCH_CACHE = pathlib.Path(pooch.os_cache("openfe_analysis"))
POOCH_CACHE.mkdir(parents=True, exist_ok=True)
DATA_URLS = {
    filename: (
        f"https://data.openfe.org/v1/"
        f"{digest.removeprefix('sha256:')}/{filename}"
    )
    for filename, digest in DATA_FILES.items()
}

RBFE_DATA = pooch.create(
    path=POOCH_CACHE,
    base_url="",
    registry=DATA_FILES,
    urls=DATA_URLS,
)

def _fetch_and_untar(dirname: str) -> pathlib.Path:
    archive = f"{dirname}.tar.gz"
    cached_dir = POOCH_CACHE / f"{archive}.untar" / dirname

    # Prevent concurrent pytest-xdist workers from downloading/extracting the same archive.
    lock_path = POOCH_CACHE / f"{archive}.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        RBFE_DATA.fetch(archive, processor=pooch.Untar())

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
def septop_data_dir() -> pathlib.Path:
    cached_dir = _fetch_and_untar("openfe_analysis_septop")
    return cached_dir


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


@pytest.fixture(scope="session")
def septop_complex_data(septop_data_dir):
    return {
        "pdb": septop_data_dir / "alchemical_system.pdb",
        "nc": septop_data_dir / "complex.nc",
        "ligand_A_indices": np.load(septop_data_dir / "ligand_A_indices.npy").tolist(),
        "ligand_B_indices": np.load(septop_data_dir / "ligand_B_indices.npy").tolist(),
        "rdmol_A": Chem.SDMolSupplier(str(septop_data_dir / "ligand_A.sdf"), removeHs=False)[0],
        "rdmol_B": Chem.SDMolSupplier(str(septop_data_dir / "ligand_B.sdf"), removeHs=False)[0],
    }
