import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from jinja2 import BaseLoader, Environment, StrictUndefined

from src.common import flatten
from src.experiment_cards.util import merge_dictionaries, set_config_at_level

logger = logging.getLogger(__name__)

JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined


@dataclass()
class ExperimentCard:
    # noinspection PyUnresolvedReferences
    """
    Base Experiment Card Describing a single experiment
    Attributes:
        name (str): Name of the experiment
        base (str): Base config from the config_directory to load with hydra.
        group (str): Group of this experiment
        overrides (Dict): List of Hydra overrides to use for the config.
        depends_on (str): The step/config this card is dependent on.
    """
    name: str
    base: str
    group: str
    description: str
    hypothesis: str
    overrides: Dict = field(default_factory=dict)
    depends_on: str = field(default=None)

    @property
    def save_name(self):
        return f"{self.group}.{self.name}"


@dataclass()
class ComposedExperiments:
    # noinspection PyUnresolvedReferences
    """
    A composition of many experiments that fall into the same group
    
    Attributes:
        name (str):  Name of the group of experiments composed.
        step_cards (Dict[str, ExperimentCard]): The list of experiment cards
            in this group.
        command_template (Optional[str]): Bash command templates for this group.
        command_kwargs (Optional[Dict]): Dictionary of arguments to pass to the 
            jinja render.
        command_fields (Optional[List]): List of config specific fields to add 
            to the command kwargs
    """
    name: str
    ablation_name: Optional[str]
    step_cards: Dict[str, ExperimentCard]
    command_template: Optional[str] = field(default=None)
    command_kwargs: Optional[Dict] = field(default_factory=dict)
    command_fields: Optional[List] = field(default_factory=list)
    description: Optional[str] = field(default=None)

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
            special_overrides = {}
            for k, v in overrides_dict.items():
                if isinstance(v, list):
                    special_overrides[k.replace('+', '')] = v
                    continue

                # Easier way to handle Hydra's override grammar as users may want
                # to put the override marker at different points.
                override_key = k
                if "++" in k:
                    override_key = f"++{k.replace('++', '')}"
                elif "+" in k:
                    override_key = f"+{k.replace('+', '')}"

                if isinstance(v, str) and "__" in v:
                    value = f'"{v}"'
                else:
                    value = v
                overrides_list.append(f"{override_key}={value}")

            logger.debug(f"{len(overrides_list)} overrides to use for {experiment.name}")
            logger.debug(f"{len(overrides_dict)} special overrides to use for {experiment.name}")
            logger.debug(f"Overrides for {experiment.name=}: {', '.join(overrides_list)}")
            save_path = output_path.joinpath(f"{experiment.save_name}.yaml")

            # Load the original configs from hydra with the overrides.
            with initialize_config_dir(config_dir=str(config_directory.absolute()),
                                       job_name="create_configs"):
                cfg = compose(config_name=experiment.base, overrides=overrides_list)

                logger.debug(f"Loaded config, now saving to {save_path}")
                with save_path.open('w', encoding='utf-8') as f:
                    # Add both the group and the name of the run to the configs
                    # before saving them. Do not use overrides for these because
                    # this is easier and it will ALWAYS occur.
                    with open_dict(cfg):
                        cfg['name'] = experiment.name
                        cfg['group'] = experiment.group
                        cfg['description'] = (
                                f"Group={self.description} | "
                                + f"{experiment.description}"
                        )
                        cfg['hypothesis'] = experiment.hypothesis
                        raw_cfg = OmegaConf.to_object(cfg)
                        for k, v in special_overrides.items():
                            raw_cfg = set_config_at_level(raw_cfg, k.split('.'), v)
                        cfg = OmegaConf.create(raw_cfg)
                    self._cfg[name] = OmegaConf.to_object(cfg)
                    f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))

    @property
    def job_name(self):
        if self.ablation_name is None:
            return self.name

        return f"{self.name}_{self.ablation_name}"

    def get_command(self, idx: int, output_path: Path):
        if not self.command_template:
            return None

        template_dict = {
            "idx"          : idx,
            "name"         : self.name,
            "ablation_name": self.ablation_name,
            "job_name"     : self.job_name,
            **self.command_kwargs
        }
        if not self._cfg:
            raise ValueError("CFG is none but trying to save command.")

        for step, experiment in self.step_cards.items():
            cfg_fields = {f: self._cfg[step][f] for f in self.command_fields}
            template_dict[step] = {
                'path'     : output_path.joinpath(f"{experiment.save_name}.yaml"),
                'save_name': experiment.save_name,
                'group'    : experiment.group,
                **cfg_fields

            }

        return self.command_template.render(**template_dict)  # type:ignore


@dataclass()
class AblationCombination:
    # noinspection PyUnresolvedReferences
    """
    An combination of multiple ablations


    NOTE: there are NOT allowed to be conflicting keys between different
    ablation groups.
    Args:
        name (str): The name of the ablation
        ablations_info (Dict): The mapping of ablation group name to the
            ablation from that group.

    Attributes:
        name (str): The name of the ablation

        overrides (Dict): The overrides for this combination.

        step_overrides(Dict): The step overrides for this combination. Step
            overrides take priority over normal overrides.
        ablation_values(Dict): The mapping of ablation group to ablation
            value name. This is NOT to be passed to init.

    """
    name: str
    overrides: Dict
    step_overrides: Dict
    ablation_values: Dict
    description: str
    hypothesis: str

    @classmethod
    def from_ablations_info(cls, name: str, ablations_info: Dict):
        overrides = {}
        step_overrides = {}
        ablation_values = {}
        ablation_descriptions = {}
        ablation_hypothesis = {}

        for k, v in ablations_info.items():
            ablation_values[k], ablation_overrides = v
            ablation_descriptions[k] = ablation_overrides.pop('description')
            hypothesis = ablation_overrides.pop('hypothesis', None)
            if hypothesis:
                ablation_hypothesis[k] = hypothesis
            try:
                overrides = merge_dictionaries(
                    overrides,
                    ablation_overrides.get('overrides', {}),
                    no_conflicting_leaves=True
                )
            except KeyError as e:
                logger.error(f"Ablation {k} in {name} has conflicting override keys")
                raise e
            try:
                step_overrides = merge_dictionaries(
                    step_overrides,
                    ablation_overrides.get('step_overrides', {}),
                    no_conflicting_leaves=True
                )
            except KeyError as e:
                logger.error(f"Ablation {k} in {name} has conflicting step overrides keys")
                raise e
        return cls(
            name=name,
            overrides=overrides,
            step_overrides=step_overrides,
            ablation_values=ablation_values,
            description=', '.join(f"{k}: {v}" for k, v in ablation_descriptions.items()),
            hypothesis=', '.join(f"{k}: {v}" for k, v in ablation_hypothesis.items())
        )

    def __eq__(self, other: 'AblationCombination') -> bool:
        if self.name != other.name:
            return False

        if self.overrides != other.overrides:
            return False

        if self.step_overrides != other.step_overrides:
            return False

        return self.ablation_values == other.ablation_values

    def get_overrides(self, step: str = None):
        if step is None:
            return self.overrides

        return merge_dictionaries(self.overrides, self.step_overrides.get(step, {}))

    @property
    def is_empty(self):
        return self.name == "NO_ABLATIONS_FOUND"


@dataclass()
class AblationGroup:
    # noinspection PyUnresolvedReferences
    """
    A group of ablations

    Attributes:
      name (str): The name of the ablation group.
      ablation_cards (Dict[str,Dict]): The dict of ablation cards
    """
    name: str
    ablation_cards: Dict[str, Dict]

    @property
    def ablation_names(self) -> List[str]:
        """
        Get the list of names for this ablation. Used for creating the ablation
        combinations.

        Returns:
            List[str]: The list of names.
        """
        return list(self.ablation_cards)

    def __getitem__(self, ablation_name):
        # Copy to make sure that the underlying mutable values can never be
        # changed by accident.
        return deepcopy(self.ablation_cards[ablation_name])

    def __setitem__(self, key, value):
        raise AttributeError("Setting an item for an ablation group is not supported")

    def __len__(self):
        return len(self.ablation_cards)

    @classmethod
    def from_ablation_dict(
            cls,
            name: str,
            ablation_group_dict: Dict[str, Dict]
    ) -> 'AblationGroup':
        ablation_cards = {}
        for ablation_name, ablation_dict in ablation_group_dict.items():
            description = ablation_dict.pop('description')
            hypothesis = ablation_dict.pop('hypothesis', None)
            step_overrides = ablation_dict.get('step_overrides', None)
            if step_overrides is not None:
                if not isinstance(step_overrides, dict):
                    raise TypeError(f"Ablation dict for {name} has step overrides of "
                                    f"type {type(step_overrides)}. Must be a dict.")
                overrides = ablation_dict.get('overrides', {})
                if not isinstance(overrides, dict):
                    raise TypeError(f"Ablation dict for {name} has overrides of "
                                    f"type {type(overrides)}. Must be a dict.")
            else:
                step_overrides = {}
                overrides = ablation_dict

            ablation_cards[ablation_name] = {
                "name"          : ablation_name,
                "description"   : description,
                "hypothesis"    : hypothesis,
                "step_overrides": step_overrides,
                "overrides"     : overrides
            }

        return cls(
            name=name,
            ablation_cards=ablation_cards
        )
