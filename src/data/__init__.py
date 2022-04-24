from .mbpp import MBPP
from .human_eval import HumanEval
from .code_search_net import CodeSearchNet
from .npv_task import NPV
from .npv_dataset_creation import *

NON_REGISTERED_TASKS = [
    "so",
    "tensorize",
    'hf_pretrain'
]
