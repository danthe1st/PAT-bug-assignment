from typing import Callable
import numpy.typing as npt

class PreprocessingInfo:
    def __init__(self, word_list: list[str], word_to_id: Callable[[str], int], word_occurences: npt.NDArray, doc_count: int):
        self.word_list = word_list
        self.word_to_id = word_to_id
        self.word_occurences = word_occurences
        self.doc_count = doc_count

class ProcessedData:
    def __init__(self, preprocessing_info: PreprocessingInfo, ids: npt.NDArray, bodies: npt.NDArray, assignees: npt.NDArray):
        self.ids=ids
        self.preprocessing_info=preprocessing_info
        self.bodies=bodies
        self.assignees=assignees

