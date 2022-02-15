import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from jinja2 import BaseLoader, Environment, StrictUndefined

from src.common import flatten

logger = logging.getLogger(__name__)

JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined


@dataclass()
class ExperimentCard:
    """
    Base Experiment Card Describing a single experiment
    """
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
    """
    A composition of many experiments that fall into the same group
    """
    name: str = field(metadata={
        "help": "Name of the group of experiments composed."
    })
    step_cards: Dict[str, ExperimentCard] = field(metadata={
        "help": "The list of experiment cards in this group."
    })
    command_template: Optional[str] = field(default=None, metadata={
        "help": "Bash command templates for this group."
    })
    command_kwargs: Optional[Dict] = field(default_factory=dict, metadata={
        "help": "Dictionary of arguments to pass to the jinja render."
    })
    command_fields: Optional[List] = field(default_factory=list, metadata={
        "help": "List of config specific fields to add to the command."
    })

    def __post_init__(self):
        if self.command_template:
            self.command_template = JINJA_ENV.from_string(self.command_template)  # type:ignore
        else:
            self.command_template = None
        self._cfg = {}

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

            overrides_dict = experiment.overrides

            # Force add the meta section that would otherwise not be there.
            if 'meta' in overrides_dict:
                overrides_dict['++meta'] = overrides_dict.pop('meta')

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
                    self._cfg[name] = OmegaConf.to_object(cfg)
                    f.write(OmegaConf.to_yaml(cfg, resolve=True))

    def get_command(self, idx: int, output_path: Path):
        if not self.command_template:
            return None

        template_dict = {
            "idx" : idx,
            "name": self.name,
            **self.command_kwargs
        }
        if not self._cfg:
            raise ValueError("CFG is none but trying to save command.")

        for step, experiment in self.step_cards.items():
            cfg_fields = {f: self._cfg[step][f] for f in self.command_fields}
            template_dict[step] = {
                'path'     : output_path.joinpath(f"{experiment.save_name}.yaml"),
                'save_name': experiment.save_name,
                **cfg_fields

            }

        return self.command_template.render(**template_dict)  # type:ignore


@dataclass()
class AblationCard:
    name: str = field(metadata={
        "help": "Name of the ablation."
    })
    overrides: Dict[str, Dict] = field(metadata={
        "help": "The key:values of the ablation."
    })


@dataclass()
class GridAblation(AblationCard):
    overrides: Dict[str, Dict] = field(metadata={
        "help": "The key values for the ablation. For the grid ablation, "
                "you must, however, specify slots to fill using jinja templates."
    })
    grid_values: Dict[str, List] = field(metadata={
        "help": "The grid of values to use."
    })

    name_template: str = field(metadata={
        'help': "The jinja template for formatting names"
    })

    def __post_init__(self):
        field_idx_mapping = {}
        all_values = []
        name_template = JINJA_ENV.from_string(self.name_template)  # type:ignore

        for i, (grid_field, grid_value) in enumerate(self.grid_values.items()):
            field_idx_mapping[i] = {}
            all_values.append(grid_value)

        new_overrides = {}
        for combo in itertools.product(*all_values):
            field_dict = {field_idx_mapping[i]: v for i, v in enumerate(combo)}
            # ablation_name =


STR_TO_CLASS_MAPPING = {
    "grid": GridAblation
}
