from importlib import resources
import pooch
import pytest


RFE_OUTPUT = pooch.create(
    path=pooch.os_cache("openfe_analysis"),
    base_url="doi:10.6084/m9.figshare.24101655",
    registry={
        "checkpoint.nc": "5af398cb14340fddf7492114998b244424b6c3f4514b2e07e4bd411484c08464",
        "db.json": "b671f9eb4daf9853f3e1645f9fd7c18150fd2a9bf17c18f23c5cf0c9fd5ca5b3",
        "hybrid_system.pdb": "07203679cb14b840b36e4320484df2360f45e323faadb02d6eacac244fddd517",
        "simulation.nc": "92361a0864d4359a75399470135f56642b72c605069a4c33dbc4be6f91f28b31",
        "simulation_real_time_analysis.yaml": "65706002f371fafba96037f29b054fd7e050e442915205df88567f48f5e5e1cf",
    }
)


RFE_OUTPUT_skipped_frames = pooch.create(
    path=pooch.os_cache("openfe_analysis"),
    base_url="doi:10.6084/m9.figshare.28263203",
    registry={
        "hybrid_system.pdb": "77c7914b78724e568f38d5a308d36923f5837c03a1d094e26320b20aeec65fee",
        "simulation.nc": "6749e2c895f16b7e4eba196261c34756a0a062741d36cc74925676b91a36d0cd",
    }
)


@pytest.fixture(scope='session')
def simulation_nc():
    return RFE_OUTPUT.fetch("simulation.nc")


@pytest.fixture(scope='session')
def simulation_skipped_nc():
    return RFE_OUTPUT_skipped_frames.fetch("simulation.nc")


@pytest.fixture(scope='session')
def hybrid_system_pdb():
    return RFE_OUTPUT.fetch("hybrid_system.pdb")


# @pytest.fixture(scope='session')
# def simulation_skipped_nc():
#     return resources.files('openfe_analysis.tests.data') / 'simulation.nc'


@pytest.fixture(scope='session')
def hybrid_system_skipped_pdb():
    return RFE_OUTPUT_skipped_frames.fetch("hybrid_system.pdb")


@pytest.fixture(scope='session')
def mcmc_serialized():
    return (
        '_serialized__class_name: LangevinDynamicsMove\n'
        '_serialized__module_name: openmmtools.mcmc\n'
        'collision_rate: !Quantity\n  unit: /picosecond\n  value: 1\n'
        'constraint_tolerance: 1.0e-06\nn_restart_attempts: 20\n'
        'n_steps: 625\nreassign_velocities: false\n'
        'timestep: !Quantity\n  unit: femtosecond\n  value: 4\n'
    )
