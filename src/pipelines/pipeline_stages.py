"""
Baseclass for pipelines
"""
import logging
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, field
from omegaconf import DictConfig
from datasets import Dataset
from transformers import PreTrainedModel, DataCollatorForSeq2Seq, PreTrainedTokenizer
import torch
from tqdm import tqdm

from src.common.config import get_device_from_cfg
from src.data import Task, load_task_from_cfg

logger = logging.getLogger(__name__)


class PipelineStage:

    def __init__(self, name: str, cfg: DictConfig):
        logger.info(f"Initializing PipelineStage '{name}'")
        self.NAME = name
        self.config = cfg

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> 'PipelineStage':
        """
        Initialization function for the stage that is to be implemented by the
        subclassed stage.

        Args:
            cfg (DictConfig): The config.

        Returns:
            The initialized PipelineStage
        """
        raise NotImplementedError()

    def __call__(self, **kwargs) -> Dict:
        """
        Execute this stage. This must be implemented by the subclass.
        """
        raise NotImplementedError()


class LoadDataStage(PipelineStage):

    def __init__(self, cfg: DictConfig, task: Task, num_proc: int = 1):
        super(LoadDataStage, self).__init__("load-data", cfg)
        self.task = task
        self.num_proc = num_proc

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> 'LoadDataStage':
        """
        Initialization function for the ``LoadDataStage``. Creates the ``Task``
        object from the config and returns the initialized stage.

        Args:
            cfg (DictConfig): The config.

        Returns:
            The initialized ``LoadDataStage``
        """
        return cls(cfg, task=load_task_from_cfg(cfg), num_proc=cfg.get('num_proc', 1))

    def __call__(self, data_path: Path, **kwargs) -> Dict[str, Dataset]:
        """
        Load data at ``data_path`` with the assigned ``Task``
        Args:
            data_path (Path): The path to the data.
            **kwargs: Other arguments passed in that are not of importance.

        Returns:
            The read datasets. Has two keys ``raw`` is the raw dataset.
            ``tokenized`` is the tokenized dataset.
        """
        raw, tokenized = self.task.read_data(data_path, self.num_proc, set_format='torch')
        return {'raw': raw, 'tokenized': tokenized}

    @property
    def tokenizer(self):
        return self.task.tokenizer


class PredictStage(PipelineStage):

    def __init__(
            self,
            cfg: DictConfig,
            generation_kwargs: Dict,
            batch_size: int,
            device: torch.device
    ):
        super(PredictStage, self).__init__("predict", cfg)
        self.generation_kwargs = generation_kwargs
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> 'PredictStage':
        """
        Initialization function for the ``PredictStage``.

        Args:
            cfg (DictConfig): The config.

        Returns:
            The initialized ``PredictStage``
        """
        return cls(
            cfg,
            generation_kwargs=cfg.get("generation", {}),
            batch_size=cfg["training"].get("batch_size", 4),
            device=get_device_from_cfg(cfg)
        )

    def __call__(
            self,
            model: PreTrainedModel,
            data: Dataset,
            task: Task,
            **kwargs
    ) -> Dict:
        """
        Generate predictions from a model for a given dataset.

        Args:
            model (PreTrainedModel): The model to use.
            data (Dataset): The dataset to use.
            task (Task): The task.
            **kwargs: Other args that are not used.

        Returns:
            Dict with ``indices``, ``predictions``, ``labels``.
        """
        collator = DataCollatorForSeq2Seq(
            tokenizer=task.tokenizer,
            pad_to_multiple_of=2,
            max_length=1024,
            padding="longest",
            label_pad_token_id=task.tokenizer.pad_token_id,
        )
        data_loader = torch.utils.data.DataLoader(
            data,
            collate_fn=collator,
            shuffle=False,
            batch_size=self.batch_size,
        )

        logger.info("Starting Generation")
        logger.info("Generation kwargs:")
        for k, v in self.generation_kwargs.items():
            logger.info(f"\t{k:>20} = {v}")

        indices = []
        predictions = []
        labels = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating"):
                generated_from_batch = model.generate(
                    inputs=batch["input_ids"].to(self.device),
                    **self.generation_kwargs,
                )

                # We need to check how many sequences we return for each sample so
                # we can adequately collect them.
                num_return_sequences = self.generation_kwargs.get("num_return_sequences", 1)

                for i in range(batch["input_ids"].shape[0]):
                    preds = task.tokenizer.batch_decode(
                        generated_from_batch[
                        i * num_return_sequences: (i + 1) * num_return_sequences
                        ],
                        skip_special_tokens=True,
                    )

                    gold = task.tokenizer.batch_decode(batch["labels"][i], skip_special_tokens=True)

                    # Only use the first returned result for basic evaluation,
                    # maybe later will be more advanced.
                    predictions.append(preds[0])
                    labels.append(gold)
                    indices.append(batch["idx"][i].item())
        logger.info("Generating finished.")
        return {
            "indices"    : indices,
            "labels"     : labels,
            "predictions": predictions
        }
