import numpy as np
import numpy.typing as npt
import dill as pkl

import sys

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

TOP_K_ASSIGNEES = 0

def main():
    if len(sys.argv)>1:
        input_file = sys.argv[1]
    else:
        input_file = "issues_bugzilla_raw.pkl"

    with open(input_file, "rb") as fh:
        issues_raw = pkl.load(fh)
    ids = issues_raw[:,0]
    bodies = issues_raw[:,1]
    assignee_names = issues_raw[:,2]
    unique_assignees, assignee_counts = np.unique(assignee_names, return_counts=True)
    print(f"loaded {len(ids)} issues with {len(unique_assignees)} unique assgnees")

    if TOP_K_ASSIGNEES > 0:
        min_assignee_count = np.min(assignee_counts[np.argpartition(assignee_counts, -TOP_K_ASSIGNEES)[-TOP_K_ASSIGNEES:]])
    else:
        min_assignee_count = 1

    assignee_indices = {}
    for i, ass in enumerate(unique_assignees):
        if assignee_counts[i] >= min_assignee_count:
            assignee_indices[ass]=i

    keep_entries = []
    assignee_vecs = np.zeros((ids.shape[0], len(assignee_names)))

    assignees = np.zeros((len(ids)), dtype=int)

    

    for i, assignee in enumerate(assignee_names):
        if assignee in assignee_indices:
            assignees[i]=assignee_indices[assignee]
            keep_entries.append(i)

    ids = np.array(ids[keep_entries], dtype=int)
    bodies = bodies[keep_entries]
    assignees = assignees[keep_entries]

    print(f"{len(ids)} issues in dataset")

    bodies = clean_body_array(bodies)


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

if __name__ == "__main__":
    main()