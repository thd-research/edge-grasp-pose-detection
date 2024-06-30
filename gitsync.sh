#!/bin/bash

DEFAULT_BRANCH="main"
DEFAULT_MESSAGE="General update from gitsync command."

# Set the default values if not provided abvoe (VARIABLE=${PARAMETER:-DEFAULT_VALUE})
BRANCH=${1:-$DEFAULT_BRANCH}
COMMIT_MESSAGE=${2:-$DEFAULT_MESSAGE}

git status
git fetch
git merge origin/$BRANCH
git add .
git commit -m "$COMMIT_MESSAGE"
git push origin $BRANCH

echo ""
echo "---------------------------------"
echo "Git cheatsheet"
echo "---------------------------------"

echo "1) See Git Log: git log"
echo "2) Open Git Configuration File (:q to quit): git config --global --edit"
echo "3) Show list of branches and highlights of current branch: git branch"
echo "4) Switch branch: git checkout <branch_name>"
echo "5) Create new branch: git checkout -b <new_branch_name>"
echo "6) Commits on top of another base tip (main branch sync): git rebase <branch_name>"
echo "---------------------------------"