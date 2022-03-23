"""
Code for handling
"""
import logging
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Dict, Iterator
from bs4 import BeautifulSoup
from bs4.element import Tag
import re
from unidecode import unidecode

logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class StackOverflowProcessor:
    def __init__(
            self,
            prompt_fn: Callable,
            answer_sorting: str = 'accepted',
            repeat_prompt_each_answer: bool = False,
            answers_per_sample: int = -1,
            top_answer_cutoff: int = 12,
            good_answer_cutoff: int = 3,
            bad_answer_cutoff: int = -1,
            remove_modality: str = "NONE",
            no_answer_str: str = "There is not an answer",
            allow_no_answer: bool = False,
            repeat_body_for_each_answer: bool = False,
            wrap_answer_character: str = None,
            date_format_str: str = "%Y",
            highest_is_best: bool = False,
            allow_negative_best_answer: bool = False,
            worst_is_best: bool = False
    ):
        self.answer_sorting = answer_sorting.lower()
        if self.answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {self.answer_sorting}")

        self.prompt_fn = prompt_fn
        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.top_answer_cutoff = top_answer_cutoff
        self.answers_per_sample = answers_per_sample
        self.repeat_prompt_each_answer = repeat_prompt_each_answer
        self.repeat_body_for_each_answer = repeat_body_for_each_answer
        self.allow_no_answer = allow_no_answer
        self.no_answer_str = no_answer_str
        self.date_format_str = date_format_str
        self.wrap_answer_character = wrap_answer_character
        self.highest_is_best = highest_is_best
        self.worst_is_best = worst_is_best
        self.allow_negative_best_answer = allow_negative_best_answer
        if wrap_answer_character:
            if wrap_answer_character.upper() in ['BLOCK', 'LINE']:
                self.wrap_answer_character = wrap_answer_character.upper()
            elif wrap_answer_character.upper() == 'NONE':
                self.wrap_answer_character = None
            else:
                raise ValueError(f"Unknown answer wrap {wrap_answer_character=}, disabling")
        if remove_modality is None:
            self.remove_modality = "NONE"
        else:
            self.remove_modality = remove_modality.upper()
            if self.remove_modality not in ['CODE', 'NL', 'NONE']:
                self.remove_modality = 'NONE'

    def clean_html_body(self, body_str, force_keep_all=False) -> List[Tag]:
        soup = BeautifulSoup(body_str, 'lxml')
        body = soup.find('body')

        out = []
        for i, tag in enumerate(body.find_all(recursive=False)):

            # Check if in code block
            if tag.name == 'pre':
                if self.remove_modality == "CODE" and not force_keep_all:
                    continue
                code = tag.find('code', recursive=False)
                if code is None:
                    code = tag
                new_tag = soup.new_tag('code')
                new_tag.string = code.text.strip()
                out.append(new_tag)
            else:
                if self.remove_modality == 'NL' and not force_keep_all:
                    continue
                nl_text = tag.text.strip().replace('"""', '\"\"\"')
                if out and out[-1].name == 'p':
                    out[-1].string = f"{out[-1].string}\n{nl_text}"
                else:
                    new_tag = soup.new_tag('p')
                    new_tag.string = nl_text
                    out.append(new_tag)

        return out

    def turn_body_into_str(self, body_tags: List[Tag]) -> str:
        out = []
        for t in body_tags:
            if not t.string:
                continue
            if t.name == 'p':
                out.append(self.wrap_nl(t.text.strip()))
            else:
                out.append(t.text.strip())
        return unidecode('\n'.join(o for o in out if o.strip()))

    def wrap_nl(self, nl_str):
        if not nl_str:
            return ''

        wrap_char = self.wrap_answer_character

        if wrap_char:
            if wrap_char == "BLOCK":
                return f'"""\n{nl_str.strip()}\n"""'
            else:
                return f"# {nl_str.strip()}"
        else:
            return nl_str.strip()

    def process_question(
            self,
            title: str,
            body: List[Tag],
            score,
            views,
            date,
            tags
    ):

        question_date = datetime.fromisoformat(date).strftime(self.date_format_str)

        return {
            "input_sequence": title,
            "question_score": score,
            "tags"          : ','.join(tags),
            "views"         : views,
            'context'       : unidecode('\n'.join(t.text.strip() for t in body if t.text.strip())),
            'question_date' : question_date
        }

    def process_answer(self, answer: List[Tag], score, date, is_best_answer):

        if not answer:
            return self.no_answer_str

        quality_str = None
        if self.good_answer_cutoff is not None and self.bad_answer_cutoff is not None:
            if is_best_answer:
                quality_str = 'BEST'
            elif score >= self.top_answer_cutoff:
                quality_str = 'GREAT'
            elif score >= self.good_answer_cutoff:
                quality_str = "GOOD"
            elif score <= self.bad_answer_cutoff:
                quality_str = "BAD"
            else:
                quality_str = "OK"

        return self.turn_body_into_str(answer), {
            'quality'    : quality_str,
            'answer_date': datetime.fromisoformat(date).strftime(self.date_format_str)
        }

    def apply_prompt(self, prompt_kwargs, is_first_answer, answer_kwargs=None):
        copy_prompt_kwargs = deepcopy(prompt_kwargs)
        answer_kwargs = answer_kwargs or {}
        if not is_first_answer and not self.repeat_body_for_each_answer:
            copy_prompt_kwargs['question'] = None

        if answer_kwargs.get('quality', None):
            copy_prompt_kwargs['quality'] = answer_kwargs['quality']

        if answer_kwargs.get('answer_date', None):
            copy_prompt_kwargs['answer_date'] = answer_kwargs['answer_date']
        return self.prompt_fn(copy_prompt_kwargs).strip()

    def __call__(self, sample: Dict) -> List[Dict]:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        sample['body'] = self.clean_html_body(sample['body'],
                                              force_keep_all=True)

        for k in sample['answers'].keys():
            sample['answers'][k]['body'] = self.clean_html_body(sample['answers'][k]['body'])

        # Do a list comprehension to eliminate the accepted answer
        accepted_answer = None
        answers = []
        highest_scoring_answer = lowest_scoring_answer = None
        highest_score = float('-inf')
        lowest_score = float('inf')
        for d in sample['answers'].values():
            if d['score'] > highest_score:
                highest_scoring_answer = d['id']
                highest_score = d['score']
            if d['score'] < lowest_score:
                lowest_scoring_answer = d['id']
                lowest_score = d['score']
            if d['id'] == accepted_answer_id and self.answer_sorting == "accepted":
                accepted_answer = d
            else:
                answers.append(d)

        # Sort the answer keys
        answers = list(sorted(
            answers,
            key=lambda ans: ans['score'],
            reverse=self.answer_sorting != 'ascending'
        ))

        # If there is an accepted answer and we are sorting by accepted, put the
        # accepted at the front of the list.
        if accepted_answer is not None and self.answer_sorting == "accepted":
            answers = [accepted_answer, *answers]

        # Create the kwargs for the prompt.
        prompt_kwargs = self.process_question(
            title=sample['title'],
            body=deepcopy(sample['body']),
            score=sample['score'],
            views=sample['views'],
            tags=sample['tags'],
            date=sample['date']
        )
        prompt_kwargs['quality'] = 'NONE' if self.repeat_prompt_each_answer else 'BEST'

        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        # Add the quality information to the answer.
        if not answers:
            if self.allow_no_answer:
                return [
                    {
                        'input' : self.apply_prompt(prompt_kwargs, True),
                        'labels': self.no_answer_str
                    }
                ]
            return []

        answers_processed = []
        for i, answer in enumerate(answers):
            if not answer['body'] and not self.allow_no_answer:
                continue
            if i >= answers_keep:
                break

            is_best_answer = False
            if self.worst_is_best:
                is_best_answer = answer['id'] == lowest_scoring_answer
            elif answer['id'] == accepted_answer_id and not self.highest_is_best:
                is_best_answer = True
            elif (
                    answer['id'] == highest_scoring_answer
                    and (accepted_answer_id == '-1' or self.highest_is_best)
            ):
                if answer['score'] >= 0:
                    is_best_answer = True
                elif self.allow_negative_best_answer:
                    is_best_answer = True

            answers_processed.append(
                self.process_answer(
                    answer['body'], answer['score'], answer['date'],
                    is_best_answer
                )
            )

        if not answers_processed:
            if not self.allow_no_answer:
                return []
            return [
                {
                    'input' : self.apply_prompt(prompt_kwargs, True),
                    'labels': self.no_answer_str
                }
            ]
        if not self.repeat_prompt_each_answer:
            return [
                {
                    'input' : self.apply_prompt(prompt_kwargs, True, ),
                    'labels': '\n'.join(d[0] for d in answers_processed)
                }
            ]

        return [
            {
                'input' : self.apply_prompt(prompt_kwargs, i == 0, d[1]),
                'labels': d[0]
            }
            for i, d in enumerate(answers_processed)
        ]
