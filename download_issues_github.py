from typing import Optional
import requests
from tqdm import tqdm
import dill as pkl

import numpy as np
import numpy.typing as npt

REPO_PATHS = ("eclipse-jdt/eclipse.jdt.core", "eclipse-jdt/eclipse.jdt.ui")
#REPO_PATHS = ["eclipse/che"]


def process_issue(issue: dict) -> Optional[tuple[str, str, str]]:
    #if issue["author_association"] == "CONTRIBUTOR":
    #    return
    id = issue["html_url"]
    body = f"{issue['title']} {issue['body']}"
    #assignees = []
    single_assignee = issue["assignee"]
    #for assignee in issue["assignees"]:
    #    assignees.append(assignee["login"])
    #if issue["user"]["login"] in assignees:
    #    return
    if "pull_request" in issue:
        return None
    if not single_assignee:
        return None
    return id, body, single_assignee["login"]

def add_issues(issues: list[tuple[str, str, str]], token, repo_path):
    for i in tqdm(range(1, 100)):
        res = requests.request("GET", f"https://api.github.com/repos/{repo_path}/issues?state=closed&per_page=100&page={i}", headers={"Authorization": f"Bearer {token}"})
        if res.status_code != 200:
            return
        data = res.json()
        if not data:
            return
        i=i+1
        for issue in data:
            result = process_issue(issue)
            if result:
                issues.append(result)

def load() -> npt.NDArray:
    i = 1
    issues: list[tuple[str, str, str]] = []
    
    with open(".token","r") as fh:
        token = fh.readline()

    for repo_path in REPO_PATHS:
        add_issues(issues, token, repo_path)
    
    return np.array(issues, dtype=object)


def main():
    issues = load()
    print(len(issues))
    with open("issues_github_raw.pkl", "wb") as fh:
        pkl.dump(issues, fh)

if __name__ == "__main__":
    main()

