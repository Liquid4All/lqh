Also for the "Get started section", recommend updating it to use uv to run lqh  and perhaps including the venv setup too so users can just copy/paste:
uv venv --python 3.12
uv pip install lqh
uv run lqh --auto ./my-task(Bare lqh --auto ./my-task will result in an error like bash: command not found: lqh unless the virtual environment is activated first, e.g. via . .venv/bin/activate)
Screenshot 2026-07-09 at 11.30.22 AM.png [8:39 AM]Then should it show just uv run lqh so that it prompts the user through the workflow since they won't yet have a ./my-task specification file?
