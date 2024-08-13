#!/bin/bash

# -------ATTENTION: Remove this block after thourough testing ------------ #
# Please DO NOT run this template directly.
# Double check each line and make sure it's consistent with your project
# setting before running the script.
# When you are ready to use the script, change the permission of this file
# to executable so that it can be picked up by Travis.
#   e.g. git update-index --chmod=+x docs/build_docs.sh
# ----------------------------------------------------------------------- #

pip install mkdocs
pip install mkdocs-git-revision-date-plugin
pip install pymdown-extensions
pip install mkdocs-material
pip install mkdocs-autorefs

git config user.name Travis;
git config user.email your@email.com;
git remote add gh-token https://${GITHUB_TOKEN}@github.<org>.com/<path/to/repo>.git;
git fetch gh-token && git fetch gh-token gh-pages:gh-pages;
mkdocs gh-deploy -v --clean --force --remote-name gh-token;
