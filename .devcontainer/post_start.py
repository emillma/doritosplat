from pathlib import Path
from subprocess import run, CompletedProcess
import re


def run_cmd(cmd) -> CompletedProcess:
    return run(cmd, shell=True, capture_output=True).stdout.decode("utf-8")


def append(path, text):
    with open(path, "a") as f:
        f.write(text + "\n")


for path in Path("/workspaces").rglob("*"):
    if path.is_dir() and ".git" in (p.name for p in path.iterdir()):
        run_cmd(f"git config --global --add safe.directory {path}")


run_cmd('git config --global user.email "emil.martens@gmail.com"')
run_cmd('git config --global user.name "Emil Martens"')
run_cmd("git config --global core.fileMode false")
run_cmd("git config --global core.autocrlf input")
run_cmd("git config --global core.eol lf")

# run_cmd('pip install -e `find /  -type d -wholename "*/vitass/hwlib"`')

bashrc = Path.home().joinpath(".bashrc")
append(bashrc, "export DISPLAY=10.53.57.55:0.0")
append(bashrc, "export LIBGL_ALWAYS_INDIRECT=1")
# append(bashrc, "conda activate dev")
