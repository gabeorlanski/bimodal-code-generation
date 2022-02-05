import logging
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Generator
import yaml
from copy import deepcopy

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
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
        experiment_card_dict: Dict,
        parent_defaults: Dict = None,
        chain: str = ""
) -> Generator[ExperimentCard, None, None]:
    """
    Recursively get the experiment cards.

    Args:
        name (str): The name of the current card/group.
        experiment_card_dict (Dict): the dict of the experiment card.
        parent_defaults (Dict): The dict of defaults from the parent.
        chain (str): The string representing the recursive chain. For debugging.

    Yields:
        Generator[ExperimentCard, None, None]: The parsed experiment card.

    """
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
    experiment_overrides = deepcopy(parent_defaults.get('overrides', {}))
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
                "group"    : experiment_card_dict.get('group', name),
                # Use the parent's name as the group for any children.
                "base"     : base_config,
                "overrides": experiment_overrides
            },
            chain=chain
        )
        for child_name, child_dict in experiment_children.items():
            child_name_to_use = child_name
            if experiment_group != name:
                child_name_to_use = f"{name}.{child_name}"
            for experiment in experiment_partial(child_name_to_use, child_dict):
                yield experiment


def load_experiment_cards_from_file(experiment_card_path: Path) -> List[ExperimentCard]:
    """
    Load the experiment cards from a given file.
    Args:
        experiment_card_path (Path): The path to the .yaml file with the
            experiment cards.

    Returns:
        List[ExperimentCard]: The list of experiment cards.
    """
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
        # Duplicate names are strictly NOT allowed.
        logger.error("Duplicate experiment card names.")

        # Get the list of duplicates.
        logger.error(
            f"Duplicated names: {', '.join([n for n, v in unique_names.items() if v > 1])}"
        )
        raise ValueError("Duplicate Experiment names.")

    return experiment_cards


def save_experiment_cards(
        experiment_cards: List[ExperimentCard],
        output_path: Path,
        config_directory: Path
):
    """
    Save the experiment cards to their configs
    Args:
        experiment_cards (List[ExperimentCard]): The list of experiment cards
            to save.
        output_path (Path): Path for saving the experiments.
        config_directory (Path): Path where the base hydra configs are.

    Returns: None

    """
    logger.info(f"Creating experiments from the base hydra configs.")
    for experiment in experiment_cards:
        logger.debug(f"Loading hydra config {experiment.base}")

        overrides_dict = flatten(experiment.overrides, sep='.')
        overrides_list = []
        for k, v in overrides_dict.items():

            # Easier way to handle Hydra's override grammar as users may want
            # to put the override marker at different points.
            override_key = k
            if "++" in k:
                override_key = f"++{k.replace('++', '')}"
            elif "+" in k:
                override_key = f"+{k.replace('+', '')}"
            overrides_list.append(f"{override_key}={v}")

        logger.info(f"{len(overrides_list)} overrides to use for {experiment.name}")
        logger.debug(f"Overrides for {experiment.name=}: {', '.join(overrides_list)}")
        save_path = output_path.joinpath(f"{experiment.save_name}.yaml")

        # Load the original configs from hydra with the overrides.
        with initialize_config_dir(config_dir=str(config_directory.absolute()),
                                   job_name="create_configs"):
            cfg = compose(config_name=experiment.base, overrides=overrides_list)

            logger.info(f"Loaded config, now saving to {save_path}")
            with save_path.open('w', encoding='utf-8') as f:
                # Add both the group and the name of the run to the configs
                # before saving them. Do not use overrides for these because
                # this is easier and it will ALWAYS occur.
                with open_dict(cfg):
                    cfg['name'] = experiment.name
                    cfg['group'] = experiment.group
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
