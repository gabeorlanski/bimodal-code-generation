Code for the paper [Evaluating How Fine-tuning on Bimodal Data Effects Code Generation](https://arxiv.org/abs/2211.07842)

### Install Instructions

1. Clone this repo with

```shell
git clone https://github.com/gabeorlanski/springresearch.git
```

2. Install the requirements with

```shell
pip install -r requirements.txt
```

3. Install these python libraries from their repositories:

* [TaskIO](https://github.com/gabeorlanski/taskio)
* [Apex](https://github.com/NVIDIA/apex)

### Configs

Configs for this project use
the [Hydra Framework](https://github.com/facebookresearch/hydra). The main
configs are located
[in the `conf` directory](https://github.com/gabeorlanski/springresearch/tree/main/conf)
. The two most important ones are
[`train_config.yaml`](https://github.com/gabeorlanski/springresearch/blob/main/conf/train_config.yaml)
and
[`eval_config.yaml`](https://github.com/gabeorlanski/springresearch/blob/main/conf/eval_config.yaml)
. They are for
[`train.py`](https://github.com/gabeorlanski/springresearch/blob/main/train.py)
and
[`evaluate.py`](https://github.com/gabeorlanski/springresearch/blob/main/evaluate.py)
respectively. Finally,
[`training_args.yaml`](https://github.com/gabeorlanski/springresearch/blob/main/conf/training_args.yaml)
are the training args that correspond to HuggingFace's
[`Seq2SeqTrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments)
. This file is loaded automatically into `train_config.yaml`.

There are a few intricacies/things of note for how configs are parsed (`.`
indicates hierarchy in yaml configs):

1. To set a batch size, set the `training.batch_size` and it will set the
   corresponding arguments for the huggingface training arguments. They
   are `per_device_train_batch_size`
   and `per_device_eval_batch_size`.
2. The `model_type` argument for `train_confing.yaml` is `seq2seq`. This will
   select HuggingFace's
3. [`AutoModelForSeq2SeqLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM)
   . This means that the model name passed to the `model` argument __must__ be a
   valid for the corresponding `model_type`. The currently
   supported `model_types` values are:
    * `seq2seq`
      --> [`AutoModelForSeq2SeqLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM)
    * `causal_lm`
      --> [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) 
    **NOTE:** This will add in a _postprocessor_ that removes the prompt/context
      from the generated output.
4. To load a checkpoint into training  (instead of starting from a HF
   checkpoint) set the argument `is_checkpoint=true`.
   
## Citation
```
@article{orlanski2022evaluating,
  title={Evaluating How Fine-tuning on Bimodal Data Effects Code Generation},
  author={Orlanski, Gabriel and Yang, Seonhye and Healy, Michael},
  journal={arXiv preprint arXiv:2211.07842},
  year={2022}
}
```
