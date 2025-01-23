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


def test_unitedyamlloader(mcmc_serialized):
    data = yaml.load(mcmc_serialized, Loader=UnitedYamlLoader)

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
