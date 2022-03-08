import json
import logging
import multiprocessing
import os
import random
import re

import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset, load_metric
from omegaconf import OmegaConf, open_dict, DictConfig
from tqdm import tqdm
import click

import transformers
from arguments import HumanEvalArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    set_seed,
)

from src.common import PROJECT_ROOT, setup_global_logging, flatten
from src.config import (
    load_model_from_cfg, setup_tracking_env_from_cfg, get_config_for_tracking,
    get_run_base_name_from_cfg, merge_configs
)
from src.evaluation.code_eval import evaluate_code

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


def complete_code(pipe, prompt, remove_prompt=False, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
    if remove_prompt:
        return [first_block(code_gen["generated_text"][len(prompt):]) for code_gen in code_gens]
    else:
        return [first_block(code_gen["generated_text"]) for code_gen in code_gens]


@click.command()
@click.argument('cfg', metavar='<Path To Config>')
@click.option('--objective', default=None, help='<Objective>')
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--notrack', 'disable_tracking',
    is_flag=True,
    default=False,
    help="Disable Tracking"
)
@click.option(
    '--no-code-eval',
    is_flag=True,
    default=False,
    help="Do Not Run Code"
)
@click.option(
    '--debug-tasks', '-tasks',
    default=None,
    type=int,
    help="# Tasks for Debugging"
)
@click.option(
    '--seq-per-sample', '-seqs', 'sequences_per_sample',
    default=None, type=int, help='Number of sequences per sample to generate.'
)
@click.option(
    '--batch-size', '-B', 'batch_size',
    default=None, type=int, help='Number of sequences per batch to generate.')
@click.option(
    '--num-workers', '-n', 'num_workers',
    default=None, type=int, help='Number of workers to use.'
)
@click.option(
    '--temperature', '-temp',
    default=None, type=float, help='Temperature to use'
)
@click.option(
    '--top-k', '-topk', 'top_k',
    default=None, type=int, help='Top-K value to use'
)
@click.option(
    '--top_p', '-topp', 'top_p',
    default=None, type=int, help='Top-P value to use'
)
def main(
        cfg,
        objective,
        debug,
        disable_tracking,
        no_code_eval,
        debug_tasks,
        batch_size,
        sequences_per_sample,
        num_workers,
        temperature,
        top_k,
        top_p
):
    # Setup configuration
    cfg = OmegaConf.create(yaml.load(
        PROJECT_ROOT.joinpath(cfg).open('r'),
        yaml.Loader
    ))

    if cfg.is_checkpoint:
        train_cfg = OmegaConf.create(yaml.load(
            PROJECT_ROOT.joinpath(cfg.model_path, 'config.yaml').open('r'),
            yaml.Loader
        ))
        cfg = merge_configs(cfg, train_cfg)

    with open_dict(cfg):
        if sequences_per_sample is not None:
            cfg.seq_per_sample = sequences_per_sample
        if batch_size is not None:
            cfg.batch_size = batch_size
        if objective is not None:
            cfg.objective = objective
        cfg.debug = debug
        if disable_tracking:
            cfg.tracking = False
        cfg.num_workers = num_workers or max(multiprocessing.cpu_count() // 2, 1)

        if temperature is not None:
            cfg.generation.temperature = temperature
        if top_k is not None:
            cfg.generation.top_k = top_k
        if top_p is not None:
            cfg.generation.top_p = top_p

        cfg.group = 'HUMAN_EVAL'
        if 'task' not in cfg or 'name' not in cfg.task:
            cfg.task = OmegaConf.create(yaml.load(
                PROJECT_ROOT.joinpath('conf', 'task', 'human_eval.yaml').open('r'),
                yaml.Loader
            ))

    working_dir = PROJECT_ROOT.joinpath(
        'eval_results', "HUMAN_EVAL",
        f"{cfg.task.name.upper()}.{cfg.name}"
    )

    if not working_dir.exists():
        working_dir.mkdir(parents=True)

    setup_global_logging(
        'evaluate',
        working_dir,
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        debug=debug,
        disable_issues_file=True
    )
    logger = logging.getLogger("evaluate")
    logger.info("Starting Evaluate")
    logger.info(f"Working directory is {working_dir}")
    logger.debug(f"{cfg.seq_per_sample} sequences to be generated per sample.")

    setup_tracking_env_from_cfg(cfg)

    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = '0' if no_code_eval else '1'
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(cfg.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    _, model = load_model_from_cfg(cfg)
    pipe = pipeline(
        "text-generation" if objective == 'lm' else 'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        device=cfg.device
    )
    logger.info(f"Using {model.device}")
    logger.info(f"Using {cfg.device=}")

    # Generation settings
    gen_kwargs = {
        "stopping_criteria": StoppingCriteriaList(
            [EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
        "do_sample"        : cfg.generation.do_sample,
        "temperature"      : cfg.generation.temperature,
        "top_p"            : cfg.generation.top_p,
        "top_k"            : cfg.generation.top_k,

    }
    if cfg.generation.get('max_length'):
        gen_kwargs['max_length'] = cfg.generation.max_length
    else:
        gen_kwargs["max_new_tokens"] = cfg.generation.max_new_tokens
    if cfg.generation.get('min_length'):
        gen_kwargs['min_length'] = cfg.generation.min_length

    logger.info("Generation Parameters:")
    for k, v in cfg.generation.items():
        logger.info(f"\t{k:>16}={v}")
    # Load evaluation dataset and metric
    logger.info(f"Loading the dataset")
    human_eval = load_dataset("openai_humaneval")['test']

    # Generate completions for evaluation set
    logger.info(f"Starting Generation")

    n_tasks = debug_tasks if debug_tasks is not None else len(human_eval["test"])
    predictions = []
    logger.info(f"{n_tasks=}")
    logger.info(f"{cfg.batch_size=}")
    logger.info(f"{cfg.seq_per_sample=}")

    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = human_eval[task]["prompt"].strip()
        gen_kwargs["stopping_criteria"][0].start_length = len(tokenizer(prompt)["input_ids"])
        for batch in range(cfg.seq_per_sample // cfg.batch_size):
            task_generations.extend(
                complete_code(
                    pipe,
                    prompt,
                    remove_prompt=objective == 'lm',
                    num_completions=cfg.batch_size,
                    **gen_kwargs
                )
            )

        test_func = human_eval[task]["test"]
        entry_point = f"check({human_eval[task]['entry_point']})"
        if objective == 'lm':
            preds = [prompt + gen for gen in task_generations]
        else:
            preds=task_generations
        predictions.append({
            "task_id"   : task,
            "prediction": preds,
            "tests"     : ["\n" + test_func + "\n" + entry_point]
        })

    logger.info(
        f"Finished generating predictions, saving to {working_dir.joinpath('predictions.jsonl')}")
    with working_dir.joinpath('test.jsonl').open('w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Evaluate completions with "code_eval" metric
    if not no_code_eval:
        logger.info(f"Evaluating {len(predictions)} Code Predictions")
        results = evaluate_code(
            predictions,
            cfg.seq_per_sample,
            cfg.num_workers,
            3.0
        )
        logger.info(f"Saving results to {working_dir.joinpath('execution_results.json')}")
        with working_dir.joinpath('execution_results.json').open('w') as f:
            json.dump(
                results,
                f,
                indent=True
            )
    else:
        results = {}

    with working_dir.joinpath('eval_config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))

    if isinstance(cfg.tracking, (dict, DictConfig)) and not no_code_eval:
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
        wandb_run = wandb.init(
            job_type='code_eval',
            name=os.getenv('WANDB_RUN_NAME'),
            id=os.getenv('WANDB_RUN_ID'),
            project=os.getenv('WANDB_PROJECT'),
            group=f"HUMAN_EVAL[execution]",
            config=get_config_for_tracking(cfg),
            entity=os.getenv('WANDB_ENTITY'),
            # id=run_id
        )
        metrics_to_log_dict = {
            'test': {
                "outcome_pcts": results['outcome_pcts'],
                **results['overview']
            }
        }

        # WandB does not like logging things from the same step at different
        # times. Hence the ugly dict.
        wandb_run.log(flatten(metrics_to_log_dict, sep='/'), step=1)

        preds_artifact = wandb.Artifact(
            f"{get_run_base_name_from_cfg(cfg, 'preds')}-{os.getenv('WANDB_RUN_ID')}",
            type='predictions')

        preds_artifact.add_file(str(
            working_dir.joinpath('test.jsonl').resolve().absolute()
        ))
        preds_artifact.add_file(
            str(working_dir.joinpath(f'eval_config.yaml').resolve().absolute()))
        wandb_run.log_artifact(preds_artifact)

        metric_artifact = wandb.Artifact(
            f"{get_run_base_name_from_cfg(cfg, 'execution')}-{os.getenv('WANDB_RUN_ID')}",
            type='execution_metrics')
        metric_artifact.add_file(str(
            working_dir.joinpath('execution_results.json').resolve().absolute()
        ))
        wandb_run.log_artifact(metric_artifact)

        wandb_run.finish()


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
