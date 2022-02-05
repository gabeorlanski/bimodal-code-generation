import logging
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Generator, Optional
import yaml
from copy import deepcopy

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from collections import Counter
from jinja2 import BaseLoader, Environment

from src.common import flatten

logger = logging.getLogger(__name__)

JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)


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
    depends_on: str = field(default=None, metadata={
        "help": "The step/config this card is dependent on."
    })

    @property
    def save_name(self):
        return f"{self.group}.{self.name}"


@dataclass()
class ComposedExperiments:
    name: str = field(metadata={
        "help": "Name of the group of experiments composed."
    })

    step_cards: Dict[str, ExperimentCard] = field(metadata={
        "help": "The list of experiment cards in this group."
    })

    command_template: Optional[str] = field(default=None, metadata={
        "help": "Bash command templates for this group."
    })

    def __post_init__(self):
        if self.command_template:
            self.command_template = JINJA_ENV.from_string(self.command_template)  # type:ignore
        else:
            self.command_template = None

    @property
    def is_single_experiment(self):
        return len(self.step_cards) == 1

    def __iter__(self):
        for step in self.step_cards:
            yield step

    def values(self):
        for value in self.step_cards.values():
            yield value

    def __len__(self):
        return len(self.step_cards)

    def save(self, output_path: Path, config_directory: Path):
        for name, experiment in self.step_cards.items():
            logger.debug(f"Saving step {name}")
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

    def get_command(self, idx: int, output_path: Path):
        if not self.command_template:
            return None

        template_dict = {
            "idx" : idx,
            "name": self.name
        }
        for step, experiment in self.step_cards.items():
            template_dict[step] = {
                'path'     : output_path.joinpath(f"{experiment.save_name}"),
                'save_name': experiment.save_name
            }

        return self.command_template.render(**template_dict)  # type:ignore


def get_experiment_card_cfg_from_dict(
        name,
        experiment_card_dict: Dict,
        global_defaults: Dict = None
) -> Generator[ComposedExperiments, None, None]:
    """
    Get the experiment cards.

    Args:
        name (str): The name of the current card/group.
        experiment_card_dict (Dict): the dict of the experiment card.
        global_defaults (Dict): The dict of defaults from the global environment.

    Yields:
        Generator[Ablation, None, None]: The parsed group of experiment cards.

    """
    if global_defaults is None:
        global_defaults = {}
    logger.info(f"Parsing Experiment Card dict named {name}")
    ablations = experiment_card_dict.get('ablations', {})
    experiment_group = experiment_card_dict.get('group')
    base_config = experiment_card_dict.get('base')
    experiment_steps = experiment_card_dict.get('steps', [])

    # Put parent's values before the children's in the overrides so that the
    # children get priority.
    experiment_overrides = deepcopy(global_defaults.get('overrides', {}))
    experiment_overrides.update(experiment_card_dict.get('overrides', {}))

    logger.info(f"Experiment {name} has group {experiment_group}")
    logger.info(f"Experiment {name} has base config {base_config}")
    logger.info(f"Experiment {name} has {len(experiment_overrides)} total overrides")
    logger.info(f"Experiment {name} has {len(ablations)} total ablations")
    logger.debug(f"Experiment {name} overrides = {experiment_overrides}")

    if not ablations and not experiment_steps:
        if experiment_group is None:
            raise ValueError(f'Experiment {name} does not have a group.')
        if base_config is None:
            raise ValueError(f'Experiment {name} does not have a base config.')
        # There are no ablations for this experiment, so just yield it.
        yield ComposedExperiments(
            name=name,
            step_cards={
                'single': ExperimentCard(
                    name=name,
                    base=base_config,
                    group=experiment_group,
                    overrides=experiment_overrides
                )
            }
        )
    else:
        # We are at a complex card, so we need to do more.
        logger.info(f"Found {len(ablations)} ablations for {name}")
        logger.info(f"Found {len(experiment_steps)} steps for {name}")

        has_ablations = True
        if not ablations:
            logger.debug(f"{name} has no ablations")
            has_ablations = False
            ablations = {"DEFAULT_VALUE_IGNORE": {}}
        has_steps = True
        if not experiment_steps:
            if experiment_group is None:
                raise ValueError(f'Experiment {name} does not have a group.')
            if base_config is None:
                raise ValueError(f'Experiment {name} does not have a base config.')
            logger.debug(f"{name} has no steps")
            has_steps = False
            experiment_steps = [{"name": name, "group": experiment_group, "base": base_config}]

        command_str = experiment_card_dict.get('commands')
        if command_str is not None:
            command_str = '\n'.join(command_str)
        for ablation_name, ablation_overrides in ablations.items():
            previous_step = {}
            composed_experiments = ComposedExperiments(
                name=ablation_name if has_ablations else name,
                step_cards={},
                command_template=command_str
            )
            for step_num, step_dict in enumerate(experiment_steps):

                # Do basic error checking.
                step_name = step_dict.get('name')
                step_group = step_dict.get('group')
                step_base = step_dict.get('base')
                if any([step_dict.get(k) is None for k in ['name', 'group', 'base']]):
                    logger.error(f"{step_base=}")
                    logger.error(f"{step_name=}")
                    logger.error(f"{step_group=}")
                    raise ValueError(f"Step {step_num} in {name} does not have correct keys")

                step_overrides = step_dict.get("overrides", {})

                card_name = name
                if has_ablations:
                    card_name += f".{ablation_name}"

                # If there are not steps, would just repeat its own name so
                # this bool stops that.
                if has_steps:
                    card_name += f".{step_name}"

                # Deepcopy here so we do not change any of the underlying
                # mutable objects on each iteration.
                card_overrides = deepcopy(experiment_overrides)

                # Ablation gets priority over experiment
                card_overrides.update(deepcopy(ablation_overrides))

                # Step gets priority over ablation.
                card_overrides.update(step_overrides)

                # Make it into a dict config so we can use interpolation.
                cfg = OmegaConf.create({
                    # Previous step is only in here so the current step can
                    # use interpolation on it.
                    "previous_step": previous_step,
                    "name"         : card_name,
                    "group"        : step_group,
                    "base"         : step_base,
                    "overrides"    : card_overrides,
                    "depends_on"   : previous_step.get('save_name')
                })
                OmegaConf.resolve(cfg)

                cfg = OmegaConf.to_object(cfg)

                # Pop the previous step key because it is no longer needed.
                cfg.pop("previous_step")

                card = ExperimentCard(**cfg)
                composed_experiments.step_cards[step_name] = card
                previous_step = {'save_name': card.save_name, **asdict(card)}
            yield composed_experiments


def load_ablation_cards_from_file(experiment_card_path: Path) -> List[ComposedExperiments]:
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
    composed_experiments = []
    for name, experiment_dict in experiments.items():
        logger.debug(f"Processing {name}")
        cards = []
        for experiment in get_experiment_card_cfg_from_dict(name, experiment_dict):
            cards.append(experiment)

        logger.info(f"Top Level {name} had {len(cards)} total experiments")
        composed_experiments.extend(cards)
    logger.info(f"{len(composed_experiments)} total experiments found.")

    # Check for duplicate group+name combos
    duplicated_names = [
        k for k, v in
        Counter(e.save_name for ablation in composed_experiments for e in ablation.values()).items()
        if v > 1
    ]
    if duplicated_names:
        # Duplicate names are strictly NOT allowed.
        logger.error("Duplicate experiment card names.")

        # Get the list of duplicates.
        logger.error(
            f"Duplicated names: {', '.join(duplicated_names)}"
        )
        raise ValueError("Duplicate Experiment names.")

    return composed_experiments


def create_experiment_bash_string(experiment_card, save_path, current_job_ids):
    current_job_ids += 1
    pre_jid = f"pre_jid{current_job_ids}"
    out = f"{pre_jid}=$(sbatch --parsable " \
          f"--job-name={experiment_card.name}_pre train.sbatch " \
          f"{save_path})\necho \"Submitted PreTrain (id=${pre_jid})\""


def save_experiment_cards(
        composed_experiments: List[ComposedExperiments],
        output_path: Path,
        config_directory: Path
):
    """
    Save the experiment cards to their configs
    Args:
        composed_experiments (List[ExperimentCard]): The list of ablation cards
            to save.
        output_path (Path): Path for saving the experiments.
        config_directory (Path): Path where the base hydra configs are.

    Returns: None

    """
    logger.info(f"Creating experiments from the base hydra configs.")

    script_fd = output_path.joinpath('experiments.sh').open('w')
    script_fd.write("#!/bin/bash\n")
    script_fd.write("# Run experiments generated by the create_experiments.py\n")

    for i, composed in enumerate(composed_experiments):
        logger.info(f"Saving {composed.name} to {output_path}")
        composed.save(output_path, config_directory)

        command = composed.get_command(idx=i, output_path=output_path)
        if command is not None:
            script_fd.write(f"\n# Command for {composed.name}\n")
            script_fd.write(command + '\n')
    script_fd.close()
