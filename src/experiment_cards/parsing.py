import itertools
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Generator, Optional, Tuple
import yaml
from copy import deepcopy

from omegaconf import OmegaConf
from collections import Counter

from src.common import PROJECT_ROOT
from src.experiment_cards import cards
from src.experiment_cards.util import merge_dictionaries

logger = logging.getLogger(__name__)


def parse_ablations(
        name: str,
        ablations: List[Dict]
) -> List[cards.AblationGroup]:
    logger.debug(f"Getting the ablations for {name}")
    if ablations is None:
        logger.info(f"No Ablations found for {name}")
        return [cards.AblationGroup(name="NO_ABLATIONS_FOUND", ablation_cards={})]

    if not isinstance(ablations, List):
        logger.error(
            f"ablations for {name} is a {type(ablations)}, must be a list."
        )
        raise TypeError(f"Ablations is not a list for {name}")
    logger.debug(f"Ablations for {name} passed the basic error checks.")
    logger.debug(f"Found {len(ablations)} potential ablation groups for {name}")

    ablation_groups = []
    for i, group_dict in enumerate(ablations):
        group_logging_str = f"{name}.ablations[{i}]"
        logger.debug(f"Parsing group {group_logging_str}")
        if not isinstance(group_dict, dict):
            logger.error(
                f"Group dict at {group_logging_str} is a "
                f"{type(group_dict)} and not a dict."
            )
            raise TypeError(f"Ablation group {i} is not a dict")
        if len(group_dict) != 1:
            logger.error(
                f"Group dict at {group_logging_str} does not have exactly "
                f"1 key."
            )
            raise ValueError(f"Group dicts for ablations must have exactly 1 key.")

        ablation_group_name, ablations_dict = next(iter(group_dict.items()))
        logger.debug(
            f"Group name for {group_logging_str}={ablation_group_name} "
            f"with {len(ablations_dict)} total ablations"
        )

        try:
            ablation_groups.append(cards.AblationGroup.from_ablation_dict(
                name=ablation_group_name,
                ablation_group_dict=ablations_dict
            ))
        except Exception as e:
            logger.error(f"An error occurred while initializing the ablations "
                         f"at {group_logging_str}")
            logger.exception(e)
            raise e

        logger.debug(f"{group_logging_str} was successfully parsed")

    logger.info(f"Found the ablation groups "
                f"{', '.join([g.name for g in ablation_groups])} for {name}")

    return ablation_groups
