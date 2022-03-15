"""
Code for handling
"""
import logging
from copy import deepcopy
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
            answer_sorting: str = 'accepted',
            answers_per_sample: int = -1,
            repeat_question_for_each_answer: str = 'none',
            top_answer_cutoff: int = 12,
            good_answer_cutoff: int = 3,
            bad_answer_cutoff: int = -1,
            answer_prompt: str = None,
            question_prompt: str = None,
            quality_prompt: str = None,
            title_prompt: str = None,
            tags_prompt: str = None,
            remove_modality: str = "NONE",
            no_answer_str: str = "There is not an answer",
            force_include_question: bool = False,
            force_include_title: bool = False,
            force_include_tags: bool = False,
            remove_body_title_repeat: bool = False,
            allow_no_answer: bool = False,
            wrap_question_character: str = None,
            wrap_answer_character: str = None
    ):
        self.answer_sorting = answer_sorting.lower()
        if self.answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {self.answer_sorting}")

        self.repeat_question_for_each_answer = repeat_question_for_each_answer
        if self.repeat_question_for_each_answer not in ['title', 'full', 'none']:
            raise ValueError(f"Invalid repeat mode: {self.repeat_question_for_each_answer}")

        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.top_answer_cutoff = top_answer_cutoff
        self.answer_prompt = answer_prompt if answer_prompt else '__ANSWER__'
        self.question_prompt = question_prompt if question_prompt else '__BODY__'
        self.title_prompt = title_prompt if title_prompt else '__TITLE__'
        self.quality_prompt = quality_prompt
        self.tags_prompt = tags_prompt
        self.answers_per_sample = answers_per_sample
        self.lm_concat_delim = '\n'
        self.wrap_question_character = wrap_question_character
        if wrap_answer_character:
            if wrap_answer_character.upper() in ['BLOCK', 'LINE']:
                self.wrap_question_character = wrap_question_character.upper()
            else:
                raise ValueError(f"Unknown Question wrap {wrap_question_character=}, disabling")

        self.wrap_answer_character = wrap_answer_character
        if wrap_answer_character:
            if wrap_answer_character.upper() in ['BLOCK', 'LINE']:
                self.wrap_answer_character = wrap_answer_character.upper()
            else:
                raise ValueError(f"Unknown answer wrap {wrap_answer_character=}, disabling")

        self.force_include_question = force_include_question
        self.force_include_title = force_include_title
        self.force_include_tags = force_include_tags
        self.remove_body_title_repeat = remove_body_title_repeat
        self.allow_no_answer = allow_no_answer

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

    def turn_body_into_str(self, body_tags: List[Tag], mode) -> str:
        out = []
        for t in body_tags:
            if not t.string:
                continue
            if t.name == 'p':
                out.append(self.wrap_nl(t.text.strip(), mode))
            else:
                out.append(t.text.strip())
        return unidecode('\n'.join(o for o in out if o.strip()))

    def wrap_nl(self, nl_str, mode):
        if not nl_str:
            return ''

        if mode == "QUESTION":
            wrap_char = self.wrap_question_character
        else:
            wrap_char = self.wrap_answer_character

        if wrap_char:
            if wrap_char == "BLOCK":
                return f'"""\n{nl_str.strip()}\n"""'
            else:
                return f"# {nl_str.strip()}"
        else:
            return nl_str.strip()

    def apply_question_prompt(
            self,
            title: str,
            body: List[Tag],
            score,
            views,
            tags,
            is_first_answer
    ):
        title_str = self.title_prompt.replace('__TITLE__', title)

        if self.tags_prompt is not None:
            # Add the extra space at the end so that it will automatically be
            # spaced out from the title.
            tags_str = self.tags_prompt.replace('__TAGS__', ' '.join(tags)) + "\n"
        else:
            tags_str = ''

        if self.remove_modality == "NL" and not self.force_include_title:
            if self.force_include_tags and tags_str:
                title_str = f"{tags_str.strip()}"
            else:
                title_str = ''
        else:
            title_str = f"{tags_str}{title_str}"

        if (
                self.repeat_question_for_each_answer == 'title' and (
                not is_first_answer
                or (is_first_answer and self.remove_body_title_repeat)
        )):
            return self.wrap_nl(title_str, "QUESTION")

        if body[0].name == 'p':
            body[0].string = f"{title_str}\n{body[0].string}"
        else:
            soup = BeautifulSoup(f"<p>{title_str}</p>", 'lxml')
            body = [soup.find('p')] + body  # type:ignore
        return self.turn_body_into_str(body, "QUESTION")

    def apply_answer_prompt(self, answer: List[Tag], score, is_accepted):

        if not answer:
            return self.no_answer_str

        quality_adjective = ""
        if self.good_answer_cutoff is not None and self.bad_answer_cutoff is not None:
            if is_accepted:
                quality_adjective = 'Accepted'
            elif score >= self.top_answer_cutoff:
                quality_adjective = 'Great'
            elif score >= self.good_answer_cutoff:
                quality_adjective = "Good"
            elif score <= self.bad_answer_cutoff:
                quality_adjective = "Bad"
            else:
                quality_adjective = "Ok"
        if self.quality_prompt:
            quality_str = self.quality_prompt.replace('__QUALITY__', quality_adjective)
        else:
            quality_str = None

        # answer_str = self.answer_prompt.replace('__ANSWER__', )
        return quality_str, self.turn_body_into_str(answer, "ANSWER")

    def make_instances_from_question(self, sample: Dict) -> List[Dict]:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        sample['body'] = self.clean_html_body(sample['body'],
                                              force_keep_all=self.force_include_question)

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
            title=sample['title'],
            body=deepcopy(sample['body']),
            score=sample['score'],
            views=sample['views'],
            tags=sample['tags'],
            is_first_answer=True
        )

        repeat_answer_input_str = self.apply_question_prompt(
            title=sample['title'],
            body=sample['body'],
            score=sample['score'],
            views=sample['views'],
            tags=sample['tags'],
            is_first_answer=False
        )

        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        # Add the quality information to the answer.
        out = []
        if not answers:
            if self.allow_no_answer:
                return [{'input': question_str, 'labels': self.no_answer_str}]
            return []

        for i, answer in enumerate(answers[:answers_keep]):
            if not answer['body'] and not self.allow_no_answer:
                continue

            quality_str, answer_str = self.apply_answer_prompt(answer['body'], answer['score'],
                                                               answer['id'] == accepted_answer_id)
            if i > 0:
                input_str = repeat_answer_input_str
            else:
                input_str = question_str

            if quality_str and self.repeat_question_for_each_answer != 'none':
                if self.wrap_question_character is None:
                    input_str += '\n' + quality_str
                elif self.wrap_question_character == "BLOCK":
                    if input_str.strip().endswith('\n"""'):
                        input_str = input_str.strip()[:-4]
                        input_str += f'\n{quality_str}\n"""'
                    else:
                        input_str += f'\n"""\n{quality_str}\n"""'
                else:
                    input_str += f"\n# {quality_str}"
            elif self.repeat_question_for_each_answer == 'none':
                if quality_str:
                    if self.wrap_answer_character is None:
                        answer_str = f"{quality_str}\n{answer_str.lstrip()}"
                    elif self.wrap_answer_character == "BLOCK":
                        if answer_str.startswith('"""'):
                            answer_str = answer_str[3:]
                            answer_str = f'"""\n{quality_str}\n{answer_str.lstrip()}'
                        else:
                            answer_str = f'"""\n{quality_str}\n"""\n{answer_str.lstrip()}'
                    else:
                        answer_str = f"# {quality_str}\n{answer_str.lstrip()}"
                out.append({'input': '', 'labels': answer_str})
                continue

            out.append({'input': input_str, 'labels': answer_str})

        if self.repeat_question_for_each_answer == 'none' and out:
            out = [{'input': question_str, 'labels': '\n'.join(d['labels'] for d in out)}]
        return out

    def __call__(self, samples):
        inputs = []
        targets = []

        for instance_list in map(self.make_instances_from_question, samples):
            for d in instance_list:
                inputs.append(d['input'])
                targets.append(d['labels'])

        return {'inputs': inputs, 'labels': targets}
