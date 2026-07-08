#!/usr/bin/env bash
rsync -avz --progress \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.lqh' \
    --exclude='e2e_results' \
    /home/mathias/dev/lqh/lqh_py/ lambda:/home/ubuntu/lqh/
