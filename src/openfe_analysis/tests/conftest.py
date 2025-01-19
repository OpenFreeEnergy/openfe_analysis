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


@pytest.fixture
def simulation_nc():
    return RFE_OUTPUT.fetch("simulation.nc")


@pytest.fixture
def hybrid_system_pdb():
    return RFE_OUTPUT.fetch("hybrid_system.pdb")


@pytest.fixture
def mcmc_serialized():
    return (
        '_serialized__class_name: LangevinDynamicsMove\n'
        '_serialized__module_name: openmmtools.mcmc\n'
        'collision_rate: !Quantity\n  unit: /picosecond\n  value: 1\n'
        'constraint_tolerance: 1.0e-06\nn_restart_attempts: 20\n'
        'n_steps: 625\nreassign_velocities: false\n'
        'timestep: !Quantity\n  unit: femtosecond\n  value: 4\n'
    )
