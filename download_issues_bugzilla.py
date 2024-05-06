import requests
from tqdm import tqdm
import dill as pkl
from typing import Optional

import numpy as np
import numpy.typing as npt

INCLUDE_BODY = False
ISSUE_COUNT = 100_000

def process_issue(issue: dict) -> Optional[tuple[str, str, str]]:
    id = issue["id"]
    body = issue["summary"]
    if INCLUDE_BODY:
        first_comment_res = requests.request("GET", f"https://bugs.eclipse.org/bugs/rest/bug/{id}/comment")
        if first_comment_res.status_code == 200:
            body += " " + first_comment_res.json()["bugs"][f"{id}"]["comments"][0]["text"]
    
    assignee = issue["assigned_to"]
    if assignee.endswith("eclipse.org"):
        return None
    return id, body, assignee

def load() -> npt.NDArray:
    issues = []
    res = requests.request("GET", f"https://bugs.eclipse.org/bugs/rest/bug?product=jdt&content=eclipse&limit={ISSUE_COUNT}&offset=0&include_fields=id,summary,assigned_to")
    assert res.status_code == 200
    data = res.json()
    assert data and "bugs" in data
    data = data["bugs"]
    for issue in tqdm(data):
        result = process_issue(issue)
        if result:
            issues.append(result)
    
    return np.array(issues, dtype=object)


def main():
    issues = load()
    print(f"{len(issues)} issues")
    with open("issues_bugzilla_raw.pkl", "wb") as fh:
        pkl.dump(issues, fh)

if __name__ == "__main__":
    main()

