import pytest
import yaml
from openfe_analysis.utils.serialization import (
    omm_quantity_string_to_offunit,
    UnitedYamlLoader,
)
from openff.units import unit


@pytest.mark.parametrize('expression, expected', [
    ['/ picosecond', 1 / unit('picosecond')],
    ['5 kilocalorie / mole', 5 * unit('kilocalorie_per_mole')],
    ['4 femtosecond', 4 * unit('femtosecond')],
])
def test_quantity_string_to_offunit(expression, expected):
    retval = omm_quantity_string_to_offunit(expression,)

    assert retval == expected


MCMC_SERIALIZED = (
    '_serialized__class_name: LangevinDynamicsMove\n'
    '_serialized__module_name: openmmtools.mcmc\n'
    'collision_rate: !Quantity\n  unit: /picosecond\n  value: 1\n'
    'constraint_tolerance: 1.0e-06\nn_restart_attempts: 20\n'
    'n_steps: 625\nreassign_velocities: false\n'
    'timestep: !Quantity\n  unit: femtosecond\n  value: 4\n'
)


def test_unitedyamlloader():
    data = yaml.load(MCMC_SERIALIZED, Loader=UnitedYamlLoader)

    expected = {
        '_serialized__class_name': 'LangevinDynamicsMove',
        '_serialized__module_name': 'openmmtools.mcmc',
        'collision_rate': 1 / unit.picosecond,
        'constraint_tolerance': 1.0e-06,
        'n_restart_attempts': 20,
        'n_steps': 625,
        'reassign_velocities': False,
        'timestep': 4 * unit.femtosecond,
    }

    assert data.keys() == expected.keys()
    for key in data.keys():
        assert data[key] == expected[key]
