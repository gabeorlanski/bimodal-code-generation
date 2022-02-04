import logging
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import List, Dict, Generator
import yaml
from collections import Counter

from src.common import flatten

logger = logging.getLogger(__name__)


@dataclass()
class ExperimentCard:
    name: str = field(metadata={
        "help": "Name of the experiment."
    })
    base: str = field(metadata={
        "help": "Base config from the config_directory to load with hydra."
    })
    group: str = field(metadata={
        "help": "Group of this experiment"
    })
    overrides: Dict = field(default_factory=dict, metadata={
        "help": "List of Hydra overrides to use for the config."
    })

    @property
    def save_name(self):
        return f"{self.group}.{self.name}"


def get_experiment_cards(
        name,
        experiment_card_dict,
        parent_defaults: Dict = None,
        chain: str = ""
) -> Generator[ExperimentCard, None, None]:
    if parent_defaults is None:
        parent_defaults = {}
    logger.info(f"Parsing Experiment Card dict named {name}")
    experiment_children = experiment_card_dict.get('experiments', {})
    experiment_group = experiment_card_dict.get('group') or parent_defaults.get('group')
    base_config = experiment_card_dict.get('base') or parent_defaults.get('base')
    logger.debug(f"Experiment {name} at {chain} has group {experiment_group}")
    logger.debug(f"Experiment {name} at {chain} has base config {base_config}")

    # Put parent's values before the children's in the overrides so that the
    # children get priority.
    experiment_overrides = parent_defaults.get('overrides', {})
    experiment_overrides.update(experiment_card_dict.get('overrides', {}))

    if not experiment_children:
        # We are at a leaf node, so yield the experiment.
        if experiment_group is None:
            raise ValueError(f'Experiment {name} at {chain} does not have a group.')
        if base_config is None:
            raise ValueError(f'Experiment {name} at {chain} does not have a base config.')
        yield ExperimentCard(
            name=name,
            base=base_config,
            group=experiment_group,
            overrides=experiment_overrides
        )
    else:
        # We are at a parent. So yield the children.
        chain = f"{chain}->{name}"
        logger.debug(f"Found {len(experiment_children)} children for {chain}")

        # Use partial here for less cluter.
        experiment_partial = partial(
            get_experiment_cards,
            parent_defaults={
                "group"    : name,  # Use the parent's name as the group for any children.
                "base"     : base_config,
                "overrides": experiment_overrides
            },
            chain=chain
        )
        for child_name, child_dict in experiment_children.items():
            for experiment in experiment_partial(child_name, child_dict):
                yield experiment


def load_experiment_cards_from_file(experiment_card_path) -> List[ExperimentCard]:
    logger.debug("Loading Experiment card")
    if not experiment_card_path.exists():
        raise FileExistsError(f"Could not find experiment card at {experiment_card_path}")

    logger.debug("Experiment card exists, loading")
    experiment_card_dict = yaml.load(
        experiment_card_path.open('r', encoding='utf-8'),
        yaml.Loader
    )
    if "experiments" not in experiment_card_dict:
        logger.error("Missing 'experiments' key")
        raise KeyError("Missing 'experiments' key")

    experiments = experiment_card_dict.get('experiments')
    logger.info(f"{len(experiments)} total top level experiments found")
    experiment_cards = []
    for name, experiment_dict in experiments.items():
        logger.debug(f"Processing {name}")
        cards = []
        for experiment in get_experiment_cards(name, experiment_dict):
            cards.append(experiment)

        logger.info(f"Top Level {name} had {len(cards)} total experiments")
        experiment_cards.extend(cards)
    logger.info(f"{len(experiment_cards)} total experiments found.")

    # Check for duplicate group+name combos
    unique_names = Counter(f"{e.group}.{e.name}" for e in experiment_cards)
    if len(unique_names) != len(experiment_cards):
        logger.error("Duplicate experiment card names.")
        # Get the list of duplicates.
        logger.error(
            f"Duplicated names: {', '.join([n for n, v in unique_names.items() if v > 1])}"
        )
        raise ValueError("Duplicate Experiment names.")

    return experiment_cards
