import yaml
from openff.units import unit
import numpy as np


def omm_quantity_string_to_offunit(expression):
    """
    Convert an OpenMM Quantity string to an OpenFF Unit.

    Parameters
    ----------
    expression : str
       The string expression to convert to an OpenFF Unit.

    Returns
    -------
    openff.units.Quantity
       An OpenFF unit Quantity.

    Notes
    -----
    Inspired by `openmmtools.utils.utils.quantity_from_string`.
    """
    # Special case where a quantity can be `/ unit` to represent `1 / unit`
    if expression[0] == '/':
        expression = f"({expression[1:]})**(-1)"

    return unit(expression)


class UnitedYamlLoader(yaml.CLoader):
    """
    A YamlLoader that can read !Quantity tags and return
    them as OpenFF Units.

    Notes
    -----
    Modified from `openmmtools.multistate.multistatereporter._DictYamlLoader`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_constructor(u'!Quantity', self.quantity_constructor)
        self.add_constructor(u'!ndarray', self.ndarray_constructor)

    @staticmethod
    def quantity_constructor(loader, node):
        loaded_mapping = loader.construct_mapping(node)
        data_unit = omm_quantity_string_to_offunit(loaded_mapping['unit'])
        data_value = loaded_mapping['value']
        return data_value * data_unit

    @staticmethod
    def ndarray_constructor(loader, node):
        loaded_mapping = loader.construct_mapping(node, deep=True)
        data_type = np.dtype(loaded_mapping['type'])
        data_shape = loaded_mapping['shape']
        data_values = loaded_mapping['values']
        data = np.ndarray(shape=data_shape, dtype=data_type)
        if 0 not in data_shape:
            data[:] = data_values
        return data
