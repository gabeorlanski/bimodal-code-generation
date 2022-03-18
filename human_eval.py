"""
This script originally came from https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot/scripts/human_eval.py
"""

import json
import logging
import math
import multiprocessing
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from functools import partial

import datasets
import jinja2
import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset, load_metric
from omegaconf import OmegaConf, open_dict, DictConfig
from tqdm import tqdm
import click
import multiprocessing as mp
import re
from tio.metrics import BLEU, ExactMatch
from apex import amp
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

from jinja2 import BaseLoader, Environment, StrictUndefined

from src.common import PROJECT_ROOT, setup_global_logging, flatten
from src.config import (
    load_model_from_cfg, setup_tracking_env_from_cfg, get_config_for_tracking,
    get_run_base_name_from_cfg, merge_configs
)
from src.evaluation.code_eval import evaluate_code

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined


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


def oracle(instance, clean_regex, metric_list):
    # Remove the block comments from the prompt.
    cleaned_target = clean_regex.sub('', instance['target'])

    # To get the oracle score, we need to repeat target for every prediction

    oracle_values = {}

    cleaned_prediction = None

    for p in instance['prediction']:

        cleaned_p = clean_regex.sub('', p)
        if cleaned_prediction is None:
            cleaned_prediction = cleaned_p

        for res in map(lambda m: m([cleaned_p], [cleaned_target]), metric_list):
            for name, value in res.items():
                oracle_values[name] = max(oracle_values.get(name, float('-inf')), value)

    return cleaned_prediction, cleaned_target, oracle_values


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
    '--move-problem',
    is_flag=True,
    default=False,
    help="Move the problem to before the signature"
)
@click.option(
    '--conditioning-tags',
    '-tags',
    default='',
    help='Comma Separated Tags to prepend the question with.', callback=lambda c, t, a: a.split(',')
)
@click.option(
    '--prompt-template',
    '-template',
    default=None,
    help='Template for preprocessing the prompt'
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
        move_problem,
        debug_tasks,
        batch_size,
        sequences_per_sample,
        num_workers,
        temperature,
        top_k,
        top_p,
        conditioning_tags,
        prompt_template
):
    torch.backends.cudnn.benchmark = True
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
        if conditioning_tags:
            cfg.group += '_COND'

        if 'task' not in cfg or 'name' not in cfg.task:
            cfg.task = OmegaConf.create(yaml.load(
                PROJECT_ROOT.joinpath('conf', 'task', 'human_eval.yaml').open('r'),
                yaml.Loader
            ))
        cfg.move_problem = move_problem

        if 'conditioning_tags' not in cfg:
            cfg.conditioning_tags = conditioning_tags or []
        elif isinstance(cfg.conditioning_tags, str):
            cfg.conditioning_tags = cfg.conditioning_tags.split(',')

        if 'prompt_template' not in cfg:
            cfg.prompt_template = prompt_template

        if cfg.prompt_template is not None:
            cfg.prompt_template = PROJECT_ROOT.joinpath(cfg.prompt_template).read_text().strip()
        else:
            cfg.prompt_template = '{{prompt}}'
        cfg.tracking.force_name = True

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

    # Generation settings
    gen_kwargs = {
        "stopping_criteria": StoppingCriteriaList(
            [EndOfFunctionCriteria(0, EOF_STRINGS, tokenizer)]),
        "do_sample"        : cfg.generation.do_sample,
        "temperature"      : cfg.generation.temperature,
        "top_p"            : cfg.generation.top_p,
        "top_k"            : cfg.generation.top_k,

    }
    if cfg.objective != 'lm':
        gen_kwargs['max_length'] = cfg.generation.get("max_length", 256)
    else:
        gen_kwargs["max_new_tokens"] = cfg.generation.get("max_new_tokens", 256)
        gen_kwargs['max_length'] = None

    if cfg.generation.get('min_length'):
        gen_kwargs['min_length'] = cfg.generation.min_length

    logger.info("Generation Parameters:")
    for k, v in cfg.generation.items():
        logger.info(f"\t{k:>16}={v}")
        if k != "stopping_criteria":
            if hasattr(model.config, k):
                setattr(model.config, k, v)
    # Load evaluation dataset and metric
    logger.info(f"Loading the dataset")
    human_eval = load_dataset("openai_humaneval")['test']

    def get_len(ex):
        ex['length'] = len(tokenizer.tokenize(ex['prompt']))
        return ex

    datasets.set_progress_bar_enabled(False)
    human_eval = human_eval.map(
        get_len,
        num_proc=cfg.num_workers,

    ).sort('length', reverse=True)

    # Generate completions for evaluation set
    logger.info(f"Starting Generation")

    n_tasks = debug_tasks if debug_tasks is not None else len(human_eval["test"])
    predictions = []
    logger.info(f"{n_tasks=}")
    logger.info(f"{cfg.batch_size=}")
    logger.info(f"{cfg.seq_per_sample=}")
    if move_problem:
        logger.info("Moving problem is enabled")
    remove_comment_regex = re.compile(r'"""(.*)"""', flags=re.DOTALL)
    start_time = datetime.utcnow()
    pipe = pipeline(
        "text-generation" if cfg.objective == 'lm' else 'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        device=cfg.device
    )
    pipe.model = pipe.model.half()
    logger.info(f"Using {model.device}")
    logger.info(f"Using {cfg.device=}")
    iteration_count, remainder = divmod(cfg.seq_per_sample, cfg.batch_size)
    has_remainder = remainder > 0
    if has_remainder:
        iteration_count += 1
    logger.info(f"{iteration_count} total iterations per task with {remainder} remainder")

    template = jinja2.Template(cfg.prompt_template)
    pbar = tqdm(total=n_tasks * math.ceil(cfg.seq_per_sample // cfg.batch_size), desc='Generating')
    for task_idx, task in enumerate(human_eval.select(list(range(n_tasks)))):
        task_generations = []
        raw_prompt = task["prompt"].strip()
        prompt = template.render(
            {'prompt': raw_prompt, 'tags': list(cfg.conditioning_tags)}
        )
        if move_problem:
            problem_statement = remove_comment_regex.search(prompt)
            if problem_statement:
                remove_comment_regex.sub('', prompt)
                prompt = f"{problem_statement.group(1).strip()}\n{prompt}"
        gen_kwargs["stopping_criteria"][0].start_length = len(tokenizer(prompt)["input_ids"])

        for batch in range(iteration_count):
            num_to_generate = cfg.batch_size
            if has_remainder and batch < iteration_count - 1:
                num_to_generate = remainder

            task_generations.extend(
                complete_code(
                    pipe,
                    prompt,
                    remove_prompt=cfg.objective == 'lm',
                    num_completions=num_to_generate,
                    **gen_kwargs
                )
            )
            pbar.update(1)

        test_func = task["test"]
        entry_point = f"check({task['entry_point']})"
        if cfg.objective == 'lm':
            preds = [raw_prompt + gen for gen in task_generations]
        else:
            preds = task_generations
        predictions.append({
            "task_id"   : task_idx,
            "prediction": preds,
            "target"    : raw_prompt + task['canonical_solution'],
            "tests"     : ["\n" + test_func + "\n" + entry_point]
        })
    pbar.close()
    total_time = (datetime.utcnow() - start_time)

    logger.info(
        f"Finished generating predictions, saving to {working_dir.joinpath('test.jsonl')}")
    with working_dir.joinpath('test.jsonl').open('w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    logger.info(f"Calculating BLEU")
    em = ExactMatch()
    bleu = BLEU()

    map_fn = partial(
        oracle,
        clean_regex=remove_comment_regex,
        metric_list=[em, bleu]
    )

    global_preds, global_targets = [], []
    global_oracle = defaultdict(list)
    logger.info(f"Calculating the Oracle Scores")
    with mp.Pool(num_workers) as pool:
        for p_cleaned, t_cleaned, oracle_scores in tqdm(
                pool.imap_unordered(map_fn, predictions), total=len(predictions)):
            global_preds.append(p_cleaned)
            global_targets.append(t_cleaned)
            for k, v in oracle_scores.items():
                global_oracle[k] = v

    metrics = {}
    metrics.update(em(global_preds, global_targets))
    metrics.update(bleu(global_preds, global_targets))

    for k, v in global_oracle.items():
        metrics[f"oracle_{k}"] = np.mean(v)

    logger.info(f"Evaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>16}={v:0.2f}")

    with working_dir.joinpath('eval_config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))

    if isinstance(cfg.tracking, (dict, DictConfig)) and not no_code_eval:
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
        group_name = f"HUMAN_EVAL{'_COND' if conditioning_tags else ''}[eval]"
        wandb_run = wandb.init(
            job_type='code_eval',
            name=os.getenv('WANDB_RUN_NAME'),
            id=os.getenv('WANDB_RUN_ID'),
            project=os.getenv('WANDB_PROJECT'),
            group=group_name,
            config=get_config_for_tracking(cfg),
            entity=os.getenv('WANDB_ENTITY'),
        )
        metrics_to_log_dict = {
            'test': {
                "runtime": total_time.total_seconds(),
                **metrics
            }
        }

        # WandB does not like logging things from the same step at different
        # times. Hence the ugly dict.
        wandb_run.log(flatten(metrics_to_log_dict, sep='/'), step=1)

        preds_artifact = wandb.Artifact(
            f"{get_run_base_name_from_cfg(cfg, 'preds')}",
            type='predictions')

        preds_artifact.add_file(str(
            working_dir.joinpath('test.jsonl').resolve().absolute()
        ))
        preds_artifact.add_file(
            str(working_dir.joinpath(f'eval_config.yaml').resolve().absolute()))
        wandb_run.log_artifact(preds_artifact)

        wandb_run.finish()


if __name__ == "__main__":
    main()
