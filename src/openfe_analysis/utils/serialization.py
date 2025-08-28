import yaml
from openff.units import unit


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
    if expression[0] == "/":
        expression = f"({expression[1:]})**(-1)"

    return unit(expression)


class UnitedYamlLoader(yaml.CLoader):
    """
    A YamlLoader that can read !Quantity tags and return
    them as OpenFF Units.

    Notes
    -----
    Modified from `openmmtools.storage.iodrivers._DictYamlLoader`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_constructor("!Quantity", self.quantity_constructor)

    @staticmethod
    def quantity_constructor(loader, node):
        loaded_mapping = loader.construct_mapping(node)
        data_unit = omm_quantity_string_to_offunit(loaded_mapping["unit"])
        data_value = loaded_mapping["value"]
        return data_value * data_unit
