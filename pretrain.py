import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
import deepspeed
import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import click
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, \
    get_scheduler, set_seed

from omegaconf import OmegaConf, open_dict
from hydra import initialize_config_dir, compose
from src.common import PROJECT_ROOT
from src.data.stackoverflow import StackOverflowProcessor
from src.config import setup_tracking_env_from_cfg, load_model_from_cfg, get_device_from_cfg, \
    get_config_for_tracking
from src.common.log_util import setup_global_logging


class ConstantLengthDataset(IterableDataset):
    def __init__(
            self,
            tokenizer,
            data_path: Path,
            processor,
            max_steps=-1,
            seq_length=1024,
            effective_batch_size=256,
            seed=1
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.data_path = data_path
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * effective_batch_size
        self.effective_batch_size = effective_batch_size
        self.epoch = 0
        self.infinite = max_steps != -1
        self.processor = processor
        if max_steps != -1:
            self.length = max_steps * effective_batch_size * seq_length
        else:
            self.length = 10

        self.rng = np.random.default_rng(seed)

    def get_next_sequence(self):
        for line in map(json.loads, self.data_path.open()):
            for instance in self.processor.make_instances_from_question(line):
                yield f"{instance['input']}\n{instance['target']}"

    def __iter__(self):
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        iterator = iter(self.get_next_sequence())
        more_examples = True
        total_yielded = 0
        while more_examples and total_yielded < self.length:
            buffer = []
            buffer_size = 0
            while buffer_size < self.max_buffer_size:
                try:
                    sequence = next(iterator)
                    sequence = self.tokenizer(sequence, truncation=False)['input_ids']
                    buffer.append(sequence)
                    buffer_size += len(sequence)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.get_next_sequence())
                        self.epoch += 1
                        print(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            self.rng.shuffle(buffer)
            all_token_ids = []
            for tokenized_input in buffer:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            slices = len(all_token_ids) // self.seq_length
            slices = slices // worker_total_num
            worker_token_slice = all_token_ids[slices * worker_id * self.seq_length:slices * (
                    worker_id + 1) * self.seq_length]
            for i in range(0, len(worker_token_slice), self.seq_length):
                input_ids = worker_token_slice[i: i + self.seq_length]
                print(f"{worker_id}: {input_ids[:5]}")
                if len(input_ids) == self.seq_length:
                    total_yielded += 1
                    yield torch.tensor(input_ids)

    def __len__(self):
        return self.length


def get_grouped_params(model, weight_decay, no_decay=None):
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def evaluate(model, eval_dataloader):
    model.eval()
    losses = []

    with torch.inference_mode():
        for step, batch in enumerate(eval_dataloader):
            local_batch = batch.to(model.device)
            outputs = model(local_batch, labels=local_batch)
            losses.append(outputs.loss.item())
            # loss = outputs.loss.repeat(1)
            # losses.append(accelerator.gather(loss))
    loss = np.mean(losses)
    try:
        perplexity = np.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss, perplexity


@click.command()
@click.argument('config_name')
@click.option('--local_rank', default=-1, type=int)
@click.option(
    '--notrack', 'disable_tracking',
    is_flag=True,
    default=False,
    help="Disable Tracking"
)
@click.option(
    '--no-deepspeed', 'disable_deepspeed',
    is_flag=True,
    default=False,
    help="Disable Deepspeed"
)
@click.option(
    '--hydra', 'use_hydra',
    is_flag=True,
    default=False,
    help="Use hydra loading for configs"
)
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option('--override-str',
              help='Bash does not like lists of variable args. so '
                   'pass as seperated list of overrides, seperated by ' '.',
              default=''
              )
def pretrain_lm(
        config_name,
        local_rank,
        disable_tracking,
        disable_deepspeed,
        debug,
        override_str,
        use_hydra
):
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
    if use_hydra:
        with initialize_config_dir(str(PROJECT_ROOT.joinpath('conf').absolute()),
                                   job_name='pretrain'):
            cfg = compose(config_name, overrides=override_str.split(" "))
    else:
        cfg = OmegaConf.create(
            yaml.load(PROJECT_ROOT.joinpath(config_name).open(), yaml.Loader)
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Load model and tokenizer
    _, model = load_model_from_cfg(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)

    # Load dataset and dataloader
    train_dataset = ConstantLengthDataset(
        tokenizer,
        PROJECT_ROOT.joinpath(cfg.train_file),
        processor=StackOverflowProcessor(**OmegaConf.to_object(cfg.processor.param)),
        max_steps=cfg.max_steps,
        seq_length=cfg.seq_length,
        effective_batch_size=cfg.train_batch_size * cfg.gradient_accumulation_steps
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        PROJECT_ROOT.joinpath(cfg.val_file),
        processor=StackOverflowProcessor(**OmegaConf.to_object(cfg.processor.param)),
        seq_length=cfg.seq_length,
        effective_batch_size=cfg.train_batch_size * cfg.gradient_accumulation_steps
    )
    eval_dataloader = DataLoader(valid_dataset, batch_size=cfg.eval_batch_size, num_workers=1)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, num_workers=1)
    device = get_device_from_cfg(cfg)
    if disable_deepspeed:
        model_engine = model.to(device)
        optimizer = AdamW(get_grouped_params(model, 0.01), lr=1e-5)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.max_steps,
        )

    else:
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=get_grouped_params(model, 0.01),  # type:ignore
            config=OmegaConf.to_object(cfg.deepspeed),
            # training_data=train_dataset
        )
        # if not disable_deepspeed:
        deepspeed.init_distributed()

    with open_dict(cfg):
        if disable_tracking:
            cfg.tracking = False

        cfg.debug = debug

    # Setup tracking
    output_dir = PROJECT_ROOT.joinpath('pretrain_outputs', f"{cfg.group}.{cfg.name}")
    checkpoint_path = output_dir.joinpath('checkpoints')
    wandb_run = None
    if local_rank <= 0:
        setup_tracking_env_from_cfg(cfg)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        else:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        checkpoint_path.mkdir()
        with output_dir.joinpath('config.yaml').open('w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

        if os.getenv("WANDB_DISABLED", 'true') == 'false':
            wandb_run = wandb.init(
                project=os.getenv('WANDB_PROJECT'),
                name=os.getenv('WANDB_RUN_NAME'),
                entity=os.getenv('WANDB_ENTITY'),
                config=get_config_for_tracking(cfg),
                id=os.getenv('WANDB_RUN_NAME'),
                group=cfg.group,
                job_type='pretrain',
            )

    def log_to_wandb(prefix, cur_step, cur_metrics):
        if local_rank <= 0 and wandb_run is not None:
            wandb_run.log({f"{prefix}/{k}": v for k, v in cur_metrics.items()}, step=cur_step)

    if not disable_deepspeed:
        torch.distributed.barrier()

    world_size = int(os.getenv("WORLD_SIZE", 1))
    setup_global_logging(
        'pretrain',
        output_dir,
        debug=debug,
        rank=local_rank,
        world_size=world_size
    )
    logger = logging.getLogger('pretrain')
    logger.info(f"Starting pretrain")

    # Train model
    model_engine.train()
    completed_steps = 0
    last_logged_step = -1
    last_eval_step = -1
    losses_since_last_update = 0

    running_loss = None
    for step, batch in enumerate(train_dataloader):
        local_batch = batch.to(model_engine.device)
        # print(batch[0,:5])
        forward_outputs = model_engine(local_batch, labels=local_batch)
        loss = forward_outputs.loss
        unscaled_loss = loss.item()
        if disable_deepspeed:
            if running_loss is None:
                running_loss = loss
            else:
                running_loss += loss
        else:
            model_engine.backward(loss)
        if step % cfg.gradient_accumulation_steps == 0:
            if disable_deepspeed and step != 0:
                running_loss /= cfg.gradient_accumulation_steps
            losses_since_last_update += running_loss.item() if disable_deepspeed else unscaled_loss

            lr_scheduler.step()
            if step != 0:
                if disable_deepspeed:
                    running_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    # weight update
                    model_engine.step()
                completed_steps += 1
            if disable_deepspeed:
                lr = optimizer.param_groups[0]["lr"]
            else:
                lr = model_engine.get_lr()[0]

            if completed_steps % cfg.logging_steps == 0 and completed_steps != last_logged_step:
                metrics = {
                    "lr"        : lr,
                    "batch_loss": running_loss.item() if disable_deepspeed else unscaled_loss
                }
                if completed_steps != 0:
                    metrics['loss'] = losses_since_last_update / cfg.logging_steps
                else:
                    metrics['loss'] = losses_since_last_update
                metrics['ds_epoch'] = train_dataloader.dataset.epoch
                logger.info(f"Step {completed_steps}: {metrics}")
                losses_since_last_update = 0
                last_logged_step = completed_steps

            log_to_wandb('train', completed_steps, metrics)

            del running_loss
            running_loss = None
            torch.cuda.empty_cache()

        if completed_steps % cfg.save_checkpoint_steps == 0 and last_eval_step != completed_steps:
            logger.info("Evaluating and saving model checkpoint")
            eval_loss, perplexity = evaluate(model_engine, eval_dataloader)
            logger.info(
                f"Eval @ Step {completed_steps}: loss={eval_loss:.4f} "
                f"perplexity={perplexity:.4f}"
            )
            log_to_wandb('eval', completed_steps, {'loss': eval_loss, 'perplexity': perplexity})
            last_eval_step = completed_steps

            # Save the checkpoint
            if completed_steps > 0 and completed_steps % cfg.save_steps == 0:
                if local_rank <= 0:
                    cur_chk = checkpoint_path.joinpath(f'checkpoint-{completed_steps}')
                    cur_chk.mkdir()
                    if disable_deepspeed:
                        model_engine.save_pretrained(cur_chk)
                    else:
                        model_engine.module.save_pretrained(cur_chk)

                if not disable_deepspeed:
                    logger.info("Waiting")
                    torch.distributed.barrier()
                    logger.info("Resuming")

        # model.train()
        if completed_steps >= cfg.max_steps:
            break

    # Evaluate and save the last checkpoint
    print("Evaluating and saving model after training")

    # Save the final model
    if local_rank <= 0:
        output_dir.joinpath('final_model').mkdir()
        if disable_deepspeed:
            model_engine.save_pretrained(output_dir.joinpath('final_model'))
        else:
            model_engine.module.save_pretrained(output_dir.joinpath('final_model'))

        with output_dir.joinpath('final_model', 'config.yaml').open('w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

        if wandb_run is not None:
            artifact = wandb.Artifact(name=f"model-{wandb_run.name}", type="model")
            artifact.add_dir(str(output_dir.joinpath('final_model').resolve().absolute()))
            if os.environ['WANDB_LOG_MODEL'] == 'true':
                wandb_run.log_artifact(artifact)
            wandb_run.finish()


if __name__ == '__main__':
    pretrain_lm()
