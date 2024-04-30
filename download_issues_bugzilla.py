import requests
from tqdm import tqdm
import dill as pkl
from typing import Optional

import numpy as np

#REPO_PATHS = ("eclipse-jdt/eclipse.jdt.core", "eclipse-jdt/eclipse.jdt.ui")
REPO_PATHS = ["eclipse/che"]


def process_issue(issue: dict) -> Optional[tuple[str, str, list[str]]]:
    #if issue["author_association"] == "CONTRIBUTOR":
    #    return
    id = issue["id"]
    #body = f"{issue['title']} {issue['body']}"
    # can get first comment with https://bugs.eclipse.org/bugs/rest/bug/1593/comment
    body = issue["summary"]
    assignee = issue["assigned_to"]
    if assignee.endswith("eclipse.org"):
        return None
    return id, body, assignee

def add_issues(issues: list[tuple[str, str, list[str]]], token, repo_path):
    limit = 100_000
    #limit = 1_000
    for i in tqdm(range(1)):
        res = requests.request("GET", f"https://bugs.eclipse.org/bugs/rest/bug?product=jdt&content=eclipse&limit={limit}&offset={i*limit}&include_fields=id,summary,assigned_to")
        if res.status_code != 200:
            return
        data = res.json()
        if not data or "bugs" not in data:
            return
        data = data["bugs"]
        for issue in data:
            result = process_issue(issue)
            if result:
                issues.append(result)

def load():
    i = 1
    issues = []
    
    with open(".token","r") as fh:
        token = fh.readline()

    for repo_path in REPO_PATHS:
        add_issues(issues, token, repo_path)
    
    return np.array(issues, dtype=object)


def main():
    issues = load()
    print(f"{len(issues)} issues")
    with open("issues_bugzilla_raw.pkl", "wb") as fh:
        pkl.dump(issues, fh)

if __name__ == "__main__":
    main()

