"""
Code for handling
"""
import logging
from typing import Callable, List, Dict, Iterator
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class StackOverflowProcessor:
    def __init__(
            self,
            answer_sorting: str = 'accepted',
            answers_per_sample: int = -1,
            repeat_question_for_each_answer: str = 'none',
            good_answer_cutoff: int = 3,
            bad_answer_cutoff: int = -1,
            answer_prompt: str = None,
            question_prompt: str = None,
            title_prompt: str = None,
            clean: bool = False
    ):
        self.answer_sorting = answer_sorting.lower()
        if self.answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {self.answer_sorting}")

        self.repeat_question_for_each_answer = repeat_question_for_each_answer
        if self.repeat_question_for_each_answer not in ['title', 'full', 'none']:
            raise ValueError(f"Invalid repeat mode: {self.repeat_question_for_each_answer}")

        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.answer_prompt = answer_prompt
        self.question_prompt = question_prompt if question_prompt else '__BODY__'
        self.title_prompt = title_prompt if title_prompt else '__TITLE__'
        self.answers_per_sample = answers_per_sample
        self.lm_concat_delim = '\n'
        self.clean = clean

    def make_instances_from_question(self, sample: Dict) -> List[Dict]:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        if self.clean:
            soup = BeautifulSoup(sample['body'], 'lxml')
            sample['body'] = soup.text

            for k in sample['answers'].keys():
                soup = BeautifulSoup(sample['answers'][k]['body'], 'lxml')
                sample['answers'][k]['body'] = soup.text

        # Do a list comprehension to eliminate the accepted answer
        accepted_answer = None
        answers = []
        for d in sample['answers'].values():
            if d['id'] == accepted_answer_id and self.answer_sorting == "accepted":
                accepted_answer = d
            else:
                answers.append(d)

        # Sort the answer keys
        answers = sorted(
            answers, key=lambda k: k['score'],
            reverse=not self.answer_sorting == 'ascending'
        )

        # If there is an accepted answer and we are sorting by accepted, put the
        # accepted at the front of the list.
        if accepted_answer is not None and self.answer_sorting == "accepted":
            answers = [accepted_answer, *answers]

        title_str = self.title_prompt.replace('__TITLE__', sample['title'])
        body_str = self.question_prompt.replace("__BODY__", sample['body'])
        question_str = f"{title_str}\n{body_str}"

        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        # Add the quality information to the answer.
        out = []
        for i, answer in enumerate(answers[:answers_keep]):

            if self.answer_prompt:
                if answer['score'] >= self.good_answer_cutoff:
                    quality_str = "good"
                elif answer['score'] <= self.bad_answer_cutoff:
                    quality_str = "bad"
                else:
                    quality_str = "ok"
                answer_str = f"{self.answer_prompt.replace('__QUALITY__', quality_str)}\n{answer['body']}"
            else:
                answer_str = answer['body']

            if i > 0:
                if self.repeat_question_for_each_answer == 'title':
                    input_str = title_str
                elif self.repeat_question_for_each_answer == 'full':
                    input_str = question_str
                else:
                    input_str = ''
            else:
                input_str = question_str

            out.append({'input': input_str, 'target': answer_str})

        if self.repeat_question_for_each_answer == 'none' and out:
            out = [{'input': out[0]['input'], 'target': '\n'.join(d['target'] for d in out)}]
        return out

    def __call__(self, samples, tokenizer):
        instances = list(map(self.make_instances_from_question, samples))
        inputs = []
        targets = []

        for instance_list in instances:
            for d in instance_list:
                inputs.append(d['input'])
                targets.append(d['target'])

        inputs_tokenized = tokenizer(inputs)
        if targets:
            targets_tokenized = tokenizer(targets)['input_ids']
        else:
            targets_tokenized = [[] for _ in range(len(inputs))]

        out = []
        for i, label in enumerate(targets_tokenized):
            out.append({
                'labels'         : label,
                'input_ids'     : inputs_tokenized['input_ids'][i],
                'attention_mask': inputs_tokenized['attention_mask'][i]
            })
        return out
