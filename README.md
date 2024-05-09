# automated Bug classification using Machine Learning methods

This repository contains code for determining the assignee of issues in Open Source projects.

## Retrieval
The issues are downloaded from <https://github.com/eclipse-jdt/eclipse.jdt.core/issues> and <https://github.com/eclipse-jdt/eclipse.jdt.ui/issues> as well as the [Eclipse Bugzilla](https://bugs.eclipse.org) using the corresponding APIs. The code for retrieving issues from GitHub is located in `download_issues_github.py` while the respective code for Bugzilla is located in `download_issues_bugzilla.py`.

The GitHub API requres an API token to be present in a `.token` file due to the amount of necessary API calls while reading issues from the Bugzilla API does not require authentication.

The endpoint in the Bugzilla API does not include the full issue description/first comment making it necessary to perform an API request for every issue. This is done when the variable `INCLUDE_BODY` is set to `True` in [`download_issues_bugzilla.py`](download_issues_bugzilla.py).
Note that doing so sends a significant amount of API requests (one for every issue as opposed to one in total) hence it is recommended to reduce `ISSUE_COUNT` (making 100000 API requests may take a while) when requesting bodies as well.

## Preprocessing
Issues are preprocessed in a script `preprocess_issues.py`.
It is possibly to supply a command-line-argument containing the name of the file containing issues downloaded by one of the aforementioned retrieval scripts.

The variable `TOP_K_ASSIGNEES` can be set in order to only consider the assignees with the most issues assigned to them.

## Classification
It is possible to train and evaluate the classifier by running `classifier.py`.