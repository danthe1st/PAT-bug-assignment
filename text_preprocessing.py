from typing import Callable

import numpy as np
import numpy.typing as npt
import re
from tqdm import trange
from shared_data import PreprocessingInfo

import torchtext
from torchtext.data import get_tokenizer

# adapted from code I wrote for the NLP course

def clean(tokenizer, text: str) -> str:
    if len(text) > 1000:
        text = text[:1000]
    #text = re.sub('[?!]', '.', text)#would normalize end of sentence but this carries too much meaning
    text = re.sub("\\d+([.,/-]\\d+)*"," 0 ",text) # reduce all numbers
    text = re.sub('[^a-zA-Z.,?!0 ]+', ' ', text) # replacing all non-letter characters except sentence deliminators with spaces
    return tokenizer(text)

def find_occurrences(cleaned_body: npt.NDArray) -> dict[str, int]:

    word_occurences: dict[str, int] = {}
    for body in cleaned_body:
        for word in body:
            if word in word_occurences:
                word_occurences[word] += 1
            else:
                word_occurences[word] = 1
    return word_occurences

def clean_dict(occurences: dict[str, int], min_occurences: int, max_occurences: int) -> list[str]:
  dictionary = ["<OOV>"]# out of vocabulary
  for word in occurences:
    if occurences[word] > min_occurences and occurences[word] < max_occurences:
      dictionary.append(word)
  return dictionary

def create_word_to_id_mapping(dictionary: list[str]) -> Callable[[str], int]:
  mapping = {}
  i = 0
  for word in dictionary:
    mapping[word]=i
    i += 1
  def map(word):
    if word in mapping:
      return mapping[word]
    else:
      return 0 # out of vocabulary
  return map

def clean_body_array(body_array: npt.NDArray):
    tokenizer = get_tokenizer("basic_english")
    body = body_array.copy()
    for i in trange(len(body)):
        body[i] = clean(tokenizer, str(body[i]))
    return body

def process_cleaned(info: PreprocessingInfo, body: npt.NDArray) -> npt.NDArray:
    processed = np.zeros((body.shape[0], len(info.word_list)))
    for i in range(body.shape[0]):
        words = body[i]
        for word in words:
            idx = info.word_to_id(word)
            processed[i,idx] = processed[i,idx]+1
    return processed

#from matplotlib import pyplot as plt

def generate_info_and_preprocess(body: npt.NDArray) -> tuple[PreprocessingInfo, npt.NDArray]:
    #body = clean_body_array(body_array)
    
    #words_flattened = np.concatenate(body).ravel()
    

    raw_dict = find_occurrences(body)
    word_list = clean_dict(raw_dict, 10, 500)

    #print(raw_dict)
    #amounts = np.array(list(raw_dict.values()))
    #amounts = amounts[amounts>10]
    #amounts = amounts[amounts<=10_000]
    #plt.hist(amounts)
    #plt.show()

    word_to_id = create_word_to_id_mapping(word_list)
    info = PreprocessingInfo(word_list, word_to_id)
    return info, process_cleaned(info, body)

def preprocess(info: PreprocessingInfo, body_array: npt.NDArray) -> npt.NDArray:
    body = clean_body_array(body_array)
    return process_cleaned(info, body)

