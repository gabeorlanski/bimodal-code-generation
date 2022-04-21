from dataclasses import dataclass, field
from typing import List, Dict


@dataclass()
class CodeSample:
    section_path: List[str]
    idx: int
    snippet_idx: int
    body_code: List[str]
    return_code: List[str]
    expected_result: List[str]
    start_char_idx: int
    context: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    testing_code: List[str] = field(default_factory=list)
    actual_returned: Dict = field(default_factory=dict)

    def aligned_returns_and_results(self):
        return zip(self.return_code, self.expected_result)

    def to_save_dict(self, annotation_mode=False):
        if annotation_mode:
            return {
                'idx'            : self.idx,
                'inputs'         : {},
                'snippet_idx'    : self.snippet_idx,
                'body_code'      : self.body_code,
                'return_code'    : self.return_code,
                'expected_result': self.expected_result,
                'context'        : '\n'.join(self.context),
            }
        return {
            'idx'            : self.idx,
            'snippet_idx'    : self.snippet_idx,
            'body_code'      : self.body_code,
            'return_code'    : self.return_code,
            'expected_result': self.expected_result,
            'start_char_idx' : self.start_char_idx,
            'context'        : '\n'.join(self.context),
            'section_path'   : '/'.join(self.section_path),
            'returned'       : self.actual_returned
        }


@dataclass()
class FailedCodeSample:
    section_name: List[str]
    idx: int
    snippet_idx: int
    error: str
    code: List[str]

    def to_dict(self):
        return {
            'idx'        : self.idx,
            'snippet_idx': self.snippet_idx,
            'error'      : self.error,
            'code'       : self.code
        }
