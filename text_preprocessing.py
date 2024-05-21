from typing import Callable

import numpy as np
import numpy.typing as npt
import re
from tqdm import trange
from shared_data import PreprocessingInfo


# adapted from code I wrote for the NLP course

def clean(text: str) -> list[str]:
    if len(text) > 1000:
        text = text[:1000]
    text = re.sub("\\d+([.,/-]\\d+)*"," 0 ",text) # reduce all numbers
    text = re.sub('[^a-zA-Z,0 ]+', ' ', text) # replacing all non-letter characters except sentence deliminators with spaces
    return text.split(' ')

def find_occurrences(cleaned_body: npt.NDArray) -> dict[str, int]:

    word_occurences: dict[str, int] = {}
    for body in cleaned_body:
        for word in body:
            if word in word_occurences:
                word_occurences[word] += 1
            else:
                word_occurences[word] = 1
    return word_occurences

def clean_dict(occurences: dict[str, int], min_occurences: int, max_occurences: int) -> tuple[list[str], npt.NDArray]:
    dictionary = ["<OOV>"]# out of vocabulary
    new_occurences = [0]
    for word in occurences:
        if occurences[word] > min_occurences and occurences[word] < max_occurences:
            dictionary.append(word)
            new_occurences.append(occurences[word])
        else:
            new_occurences[0] += 1 # increment <OOV>
    return dictionary, np.array(new_occurences)

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

def clean_body_array(body_array: npt.NDArray) -> npt.NDArray:
    body = body_array.copy()
    for i in trange(len(body)):
        body[i] = clean(str(body[i]))
    return body

def process_cleaned(info: PreprocessingInfo, body: npt.NDArray) -> npt.NDArray:
    processed = np.zeros((body.shape[0], len(info.word_list)))

    doc_lens = np.zeros(body.shape[0])

    for i, words in enumerate(body):
        for word in words:
            idx = info.word_to_id(word)
            processed[i,idx] = processed[i,idx]+1
        doc_lens[i]=len(words)
    

    doc_lens = np.repeat(doc_lens[:,np.newaxis],len(info.word_list),axis=1)
    doc_lens = np.maximum(doc_lens, 2) # avoid errors in cases of bugreports with only single word
    tf = np.log2(1+processed)/np.log2(doc_lens)
    idf = np.log2(info.doc_count/info.word_occurences+1)
    
    tfidf = tf*idf
    return tfidf

#from matplotlib import pyplot as plt

def generate_info_and_preprocess(body: npt.NDArray) -> tuple[PreprocessingInfo, npt.NDArray]:
    #body = clean_body_array(body_array)
    
    #words_flattened = np.concatenate(body).ravel()
    

    raw_dict = find_occurrences(body)

    word_list, word_occurences = clean_dict(raw_dict, 5, 500)

    #print(raw_dict)
    #amounts = np.array(list(raw_dict.values()))
    #amounts = amounts[amounts>10]
    #amounts = amounts[amounts<=10_000]
    #plt.hist(amounts)
    #plt.show()

    word_to_id = create_word_to_id_mapping(word_list)
    info = PreprocessingInfo(word_list, word_to_id, word_occurences, len(body))
    return info, process_cleaned(info, body)

def preprocess(info: PreprocessingInfo, body_array: npt.NDArray) -> npt.NDArray:
    body = clean_body_array(body_array)
    return process_cleaned(info, body)

