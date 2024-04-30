import numpy as np
import numpy.typing as npt
import dill as pkl

from text_preprocessing import generate_info_and_preprocess, clean_body_array, process_cleaned
from shared_data import PreprocessingInfo, ProcessedData

from sklearn.model_selection import train_test_split

def find_assignee_counts(assignees_list: list[list[str]]) -> dict[str, int]:
    assignee_counts: dict[str, int] = dict()
    for assignees in assignees_list:
        for assignee in assignees:
            if assignee in assignee_counts:
                cnt = assignee_counts[assignee]
            else:
                cnt = 0
            cnt = cnt+1
            assignee_counts[assignee] = cnt
    return assignee_counts
    
def get_assignee_mapping_and_names(assignees_list: list[list[str]], min_count: int = 10) -> tuple[dict[str, int], list[str]]:
    assignee_counts = find_assignee_counts(assignees_list)
    current_index = 0
    assignee_mapping = dict()
    assignee_names = []
    for assignee in assignee_counts:
        count = assignee_counts[assignee]
        if count >= min_count:
            assignee_mapping[assignee]= current_index
            assignee_names.append(assignee)
            current_index = current_index + 1
    return assignee_mapping, assignee_names

def one_hot_encoded_assignees(assignees: list[str], assignee_mapping: dict[str, int]) -> tuple[np.array, bool]:
    assignee_vec = np.zeros(len(assignee_mapping))
    has_entries = False
    for assignee in assignees:
        if assignee in assignee_mapping:
            assignee_vec[assignee_mapping[assignee]] = 1
            has_entries = True
    return assignee_vec, has_entries

def main():
    with open("issues_bugzilla_raw.pkl", "rb") as fh:
        issues_raw = pkl.load(fh)
    ids = issues_raw[:,0]
    bodies = issues_raw[:,1]
    assignee_names = issues_raw[:,2]
    unique_assignees = np.unique(assignee_names)

    assignee_indices = {}
    for i, ass in enumerate(unique_assignees):
        assignee_indices[ass]=i

    keep_entries = []
    assignee_vecs = np.zeros((ids.shape[0], len(assignee_names)))

    #assignees = np.zeros((len(ids), len(unique_assignees)), dtype=int)
    assignees = np.zeros((len(ids)), dtype=int)

    for i, assignee in enumerate(assignee_names):
        #assignees[i,assignee_indices[assignee]]=1
        assignees[i]=assignee_indices[assignee]
        keep_entries.append(i)
        #assignee_vec, has_entries = one_hot_encoded_assignees(assignees, [assignee_mapping])
        #if has_entries:
        #    keep_entries.append(i)
        #assignee_vecs[i] = assignee_vec
    
    #ids = np.array(ids[keep_entries], dtype=int)
    #bodies = bodies[keep_entries]
    bodies = clean_body_array(bodies)
    #assignees = np.array(assignee_vecs[keep_entries], dtype=int)


    train_ids, test_ids, train_bodies, test_bodies, train_assignees, test_assignees = train_test_split(ids, bodies, assignees, test_size=0.25, random_state=1337)

    preprocessing_info, train_bodies = generate_info_and_preprocess(train_bodies)
    print(f"{len(preprocessing_info.word_list)} words in dictionary")
    test_bodies = process_cleaned(preprocessing_info, test_bodies)
    data = ProcessedData(preprocessing_info, train_ids, train_bodies, train_assignees)
    with open("train.pkl", "wb") as fh:
        pkl.dump(data, fh)
    data = ProcessedData(preprocessing_info, test_ids, test_bodies, test_assignees)
    with open("test.pkl", "wb") as fh:
        pkl.dump(data, fh)
    
class ProcessedData:
    def __init__(self, preprocessing_info: PreprocessingInfo, ids: npt.NDArray, bodies: npt.NDArray, assignees: npt.NDArray):
        self.ids=ids
        self.preprocessing_info=preprocessing_info
        self.bodies=bodies
        self.assignees=assignees

if __name__ == "__main__":
    main()