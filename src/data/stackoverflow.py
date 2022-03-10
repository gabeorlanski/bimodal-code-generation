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
            clean: bool = True,
            remove_modality: str = "NONE",
            no_answer_str: str = "There is not an answer",
            force_include_question: bool = False,
            force_include_title: bool = False
    ):
        self.answer_sorting = answer_sorting.lower()
        if self.answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {self.answer_sorting}")

        self.repeat_question_for_each_answer = repeat_question_for_each_answer
        if self.repeat_question_for_each_answer not in ['title', 'full', 'none']:
            raise ValueError(f"Invalid repeat mode: {self.repeat_question_for_each_answer}")

        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.answer_prompt = answer_prompt if answer_prompt else '__ANSWER__'
        self.question_prompt = question_prompt if question_prompt else '__BODY__'
        self.title_prompt = title_prompt if title_prompt else '__TITLE__'
        self.answers_per_sample = answers_per_sample
        self.lm_concat_delim = '\n'
        self.clean = clean

        self.force_include_question = force_include_question
        self.force_include_title = force_include_title

        self.no_answer_str = no_answer_str
        if remove_modality is None:
            self.remove_modality = "NONE"
        else:
            self.remove_modality = remove_modality.upper()
            if self.remove_modality not in ['CODE', 'NL', 'NONE']:
                self.remove_modality = 'NONE'

        if force_include_title:
            if self.remove_modality is None:
                logger.warning(f"Force include title is enabled but will have no effect.")
        if force_include_question and self.remove_modality is None:
            logger.warning(f"Force include question is enabled but will have no effect.")

    def clean_html_body(self, body):
        if not self.clean and self.remove_modality == 'NONE':
            return body

        soup = BeautifulSoup(body, 'lxml').find('body')

        if self.clean and self.remove_modality == "NONE":
            return soup.text.strip()

        tag_name = 'pre' if self.remove_modality == "CODE" else 'p'

        # Remove the tags from the BS4 doc
        for t in soup.find_all(tag_name):
            t.extract()

        if self.clean:
            return soup.text.strip()

        # Do not want to clean the doc, but do not want the <body> tag. So need
        # to manually construct.
        return '\n'.join(map(repr, soup.find_all(recursive=False))).strip()

    def apply_question_prompt(self, title, body, score, views, is_first_answer):
        title_str = self.title_prompt.replace('__TITLE__', title)
        body_str = self.question_prompt.replace("__BODY__", body)

        if self.remove_modality == "NL" and not self.force_include_title:
            title_str = ''
        else:
            title_str = f"{title_str}\n"

        if is_first_answer or self.repeat_question_for_each_answer == 'full':
            return f"{title_str}{body_str}"
        if self.repeat_question_for_each_answer == 'title':
            return title_str.strip()
        return ''

    def apply_answer_prompt(self, answer, score):
        if score >= self.good_answer_cutoff:
            quality_str = "good"
        elif score <= self.bad_answer_cutoff:
            quality_str = "bad"
        else:
            quality_str = "ok"
        answer_str = self.answer_prompt.replace('__ANSWER__', answer)
        return answer_str.replace('__QUALITY__', quality_str)

    def make_instances_from_question(self, sample: Dict) -> List[Dict]:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        if self.force_include_question:
            if self.clean:
                soup = BeautifulSoup(sample['body'], 'lxml').find('body')
                sample['body'] = soup.text.strip()
        else:
            sample['body'] = self.clean_html_body(sample['body'])

        for k in sample['answers'].keys():
            sample['answers'][k]['body'] = self.clean_html_body(sample['answers'][k]['body'])

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

        question_str = self.apply_question_prompt(
            sample['title'],
            sample['body'],
            sample['score'],
            sample['views'],
            True
        )

        repeat_answer_input_str = self.apply_question_prompt(
            sample['title'],
            sample['body'],
            sample['score'],
            sample['views'],
            False
        )

        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        # Add the quality information to the answer.
        out = []
        if not answers:
            return [{'input': question_str, 'labels': self.no_answer_str}]

        for i, answer in enumerate(answers[:answers_keep]):
            answer_str = self.apply_answer_prompt(answer['body'], answer['score'])
            if i > 0:
                input_str = repeat_answer_input_str
            else:
                input_str = question_str

            out.append({'input': input_str, 'labels': answer_str})

        if self.repeat_question_for_each_answer == 'none' and out:
            out = [{'input': out[0]['input'], 'labels': '\n'.join(d['labels'] for d in out)}]
        return out

    def __call__(self, samples):
        inputs = []
        targets = []

        for instance_list in map(self.make_instances_from_question, samples):
            for d in instance_list:
                inputs.append(d['input'])
                targets.append(d['labels'])

        return {'inputs': inputs, 'labels': targets}
