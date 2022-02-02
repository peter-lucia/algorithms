#!/bin/bash
# ghp-import --help
# Usage: ghp-import [OPTIONS] DIRECTORY
# 
# Options:
#   -n, --no-jekyll       Include a .nojekyll file in the branch.
#   -c CNAME, --cname=CNAME
#                         Write a CNAME file with the given CNAME.
#   -m MESG, --message=MESG
#                         The commit message to use on the target branch.
#   -p, --push            Push the branch to origin/{branch} after committing.
#   -x PREFIX, --prefix=PREFIX
#                         The prefix to add to each file that gets pushed to the
#                         remote. Only files below this prefix will be cleared
#                         out. [none]
#   -f, --force           Force the push to the repository.
#   -o, --no-history      Force new commit without parent history.
#   -r REMOTE, --remote=REMOTE
#                         The name of the remote to push to. [origin]
#   -b BRANCH, --branch=BRANCH
#                         Name of the branch to write to. [gh-pages]
#   -s, --shell           Use the shell when invoking Git. [False]
#   -l, --follow-links    Follow symlinks when adding files. [False]
#   --version             show program's version number and exit
#   -h, --help            show this help message and exit

# -n: no jekyll
# -p: push the branch to origin/{branch}, defaults to gh-pages
# -f: force push to the repository
ghp-import -n -p -f book/_build/html
