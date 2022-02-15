import copy
import json
import math
import os
from collections import defaultdict

from datetime import datetime
from pathlib import Path
import random

import numpy as np
import wandb
from datasets import set_caching_enabled
from omegaconf import DictConfig, open_dict, OmegaConf
from transformers import PreTrainedModel, DataCollatorForSeq2Seq, StoppingCriteria, \
    StoppingCriteriaList, pipeline
import torch
import logging
from tqdm import tqdm
import re
from src.config import get_device_from_cfg, load_task_from_cfg, get_config_for_tracking, \
    get_run_base_name_from_cfg

logger = logging.getLogger(__name__)
EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(EOF_STRINGS), string)[0].rstrip()


def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
    return [code_gen["generated_text"][len(prompt):] for code_gen in code_gens]


def generate_predictions(
        model,
        split,
        task,
        device,
        generation_kwargs,
        seq_per_sample,
        debug_samples=None,
        add_back_prompt: bool = False
):
    pipe = pipeline("text-generation", model=model, tokenizer=task.tokenizer, device=device)

    logger.info(f"{pipe.device=}")
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    generations = []
    references = []
    indices = []
    num_seq = generation_kwargs.pop('num_return_sequences', 1)
    dataset = task.preprocess(split)
    task.preprocessed_splits[split] = dataset
    pbar = tqdm(total=(len(dataset) * (seq_per_sample // num_seq)))
    for i, sample in enumerate(dataset):
        if debug_samples and i > debug_samples:
            break
        task_generations = []
        prompt = sample['input_sequence'].strip()
        if 'stopping_criteria' in generation_kwargs:
            generation_kwargs["stopping_criteria"][0].start_length = len(
                task.tokenizer(prompt)["input_ids"]
            )
        for batch in range(seq_per_sample // num_seq):
            task_generations.extend(
                complete_code(
                    pipe,
                    prompt,
                    num_completions=num_seq,
                    **generation_kwargs
                )
            )
            pbar.update()

        if add_back_prompt:
            generations.append([prompt + '\n\t' + gen for gen in task_generations])
            references.append(prompt + sample['target'])
        else:
            references.append(sample['target'])
            generations.append(task_generations)
        indices.append(i)
    pbar.close()
    return {
        "indices"    : indices,
        "labels"     : references,
        "predictions": generations
    }


def evaluate_model(
        cfg: DictConfig,
        model: PreTrainedModel
):
    """
    Evaluate a model with a reader on a file
    Args:
        cfg (DictConfig): The config to use.
        model (PreTrainedModel): The pretrained huggingface model to use.
    """
    task = load_task_from_cfg(cfg)
    logger.info(f"Reading data from '{cfg['data_path']}'")

    gen_kwargs = {
        "do_sample"           : True,
        "temperature"         : 0.2,
        "max_new_tokens"      : 256,
        "top_p"               : 0.95,
        "top_k"               : 0,
        "num_return_sequences": cfg.generation.get('num_return_sequences', 1)
        # "stopping_criteria": StoppingCriteriaList([EndOfFunctionCriteria(0, EOF_STRINGS, task.tokenizer)]),
    }
    if cfg.objective == 'lm':
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token = task.tokenizer.eos_token
        model.config.eos_token_id = task.tokenizer.eos_token_id
        model.config.pad_token_id = task.tokenizer.pad_token_id
        model.config.bos_token_id = task.tokenizer.bos_token_id or task.tokenizer.eos_token
        if cfg.task.name == 'human_eval':
            gen_kwargs.update({
                "stopping_criteria": StoppingCriteriaList(
                    [EndOfFunctionCriteria(0, EOF_STRINGS, task.tokenizer)]),
            })
            task.postprocessors.append(first_block)

    # tokenized = task.get_split(cfg['split'], overwrite_cache=True)
    # logger.info(f"{len(tokenized)} total samples found")

    device = get_device_from_cfg(cfg)
    logger.info(f"Using device {device}")

    generation_results = generate_predictions(
        model.to(device),
        split=cfg['split'],
        task=task,
        device=cfg.get('device'),
        generation_kwargs=gen_kwargs,
        seq_per_sample=cfg.get('seq_per_sample'),
        debug_samples=cfg.get('debug_num_samples', None),
        # add_back_prompt=cfg.task.name == 'human_eval'
    )

    labels = generation_results['labels']
    predictions = generation_results['predictions']
    indices = generation_results['indices']

    metrics = task.evaluate(predictions, labels)
    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.3f}")

    serialized_predictions = []
    serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
    for serialized_dict in tqdm(serialize_generator, total=len(indices), desc="Serializing"):
        serialized_predictions.append(serialized_dict)

    return metrics, serialized_predictions


def evaluate(
        cfg,
        model,
        splits: str,
        out_path: Path,
        dry_run: bool,
):
    logger.debug("Setting the seeds")
    seed = cfg["seed"]
    numpy_seed = cfg["numpy_seed"]
    torch_seed = cfg["pytorch_seed"]
    logger.info(f"Seed={seed}")
    logger.info(f"NumPy Seed={numpy_seed}")
    logger.info(f"Torch Seed={torch_seed}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["numpy_seed"])
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    logger.debug(f"Starting eval loop")
    start_time = datetime.utcnow()

    logger.info(f"Using split '{splits}' for task '{cfg.task.name}'")
    splits_to_use = cfg.splits

    pred_dir = Path(out_path).joinpath('predictions')
    if not pred_dir.exists():
        pred_dir.mkdir()
    all_metrics = {}
    split_paths = []

    set_caching_enabled(not cfg.get('disable_cache', False))

    if not dry_run:
        for split in splits_to_use:
            logger.info(f"Evaluating split {split}")
            with open_dict(cfg):
                cfg.split = split
            metrics, predictions = evaluate_model(
                copy.deepcopy(cfg),
                model=model
            )

            all_metrics.update({f"{split}/{k}": v for k, v in metrics.items()})
            split_path = pred_dir.joinpath(f'{cfg.split}.jsonl')
            split_paths.append(split_path)
            logger.info(f"Saving predictions to '{split_path}'")
            with split_path.open("w", encoding="utf-8") as f:
                for serialized_dict in predictions:
                    f.write(json.dumps(serialized_dict) + '\n')

    end_time = datetime.utcnow() - start_time
    logger.info(f"Total time spent on evaluation: {end_time}")
    all_metrics['runtime'] = str(end_time)
    if not dry_run:
        with out_path.joinpath('eval_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(all_metrics, f)

    run_id = os.getenv('WANDB_RUN_ID')
    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.split = splits
        cfg.eval_run_name = os.getenv('WANDB_RUN_NAME')

    #####################################################################
    # TRACKING CODE TO REMOVE ON RELEASE                                #
    #####################################################################

    with out_path.joinpath(f'eval_config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    if (
            isinstance(cfg.tracking, (dict, DictConfig))
            and int(os.environ.get("LOCAL_RANK", "-1")) <= 0
    ):
        run = wandb.init(
            job_type='evaluate',
            name=os.getenv('WANDB_RUN_NAME'),
            project=os.getenv('WANDB_PROJECT'),
            group=f"{cfg.group}[eval]",
            entity=os.getenv('WANDB_ENTITY'),
            config=get_config_for_tracking(cfg),
            id=run_id
        )

        run.config.update(get_config_for_tracking(cfg))

        if dry_run and out_path.joinpath('eval_metrics.json').exists():
            all_metrics = json.loads(out_path.joinpath('eval_metrics.json').read_text('utf-8'))
            print(all_metrics)
        run.log({f"eval/{k}": v for k, v in all_metrics.items()}, step=1)
        preds_artifact = wandb.Artifact(get_run_base_name_from_cfg(cfg, "preds"),
                                        type='predictions')

        preds_artifact.add_dir(str(pred_dir.resolve().absolute()))
        preds_artifact.add_file(
            str(out_path.joinpath(f'eval_config.yaml').resolve().absolute()))
        run.log_artifact(preds_artifact)
        run.finish()
    logger.info("Finished Evaluation")
