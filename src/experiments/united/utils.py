from src.examples.botnet.botnet_augmented_constraints import BotnetAugmentedConstraints
from src.examples.botnet.botnet_constraints import BotnetConstraints
from src.examples.lcld.lcld_augmented_constraints import LcldAugmentedConstraints
from src.examples.lcld.lcld_constraints import LcldConstraints
from src.examples.lcld.lcld_constraints_sat import (
    create_constraints as lcld_sat_constraints,
)
from src.examples.botnet.botnet_constraints_sat import (
    create_constraints as botnet_sat_constraints,
)

STR_TO_CONSTRAINTS_CLASS = {
    "lcld": LcldConstraints,
    "botnet": BotnetConstraints,
    "lcld_augmented": LcldAugmentedConstraints,
    "botnet_augmented": BotnetAugmentedConstraints
}

STR_TO_SAT_CONSTRAINTS = {
    "lcld": lcld_sat_constraints,
    "botnet": botnet_sat_constraints,
}


def get_constraints_from_str(project_name: str):
    return STR_TO_CONSTRAINTS_CLASS[project_name]


def get_sat_constraints_from_str(project_name: str):
    return STR_TO_SAT_CONSTRAINTS[project_name]
