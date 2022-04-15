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