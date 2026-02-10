import subprocess
import os

env = os.environ.copy()
env["PYTHONPATH"] = "."

with open("traceback.log", "w") as f:
    result = subprocess.run(["python", "src/train_and_evaluate_all.py"], env=env, capture_output=True, text=True)
    f.write(result.stdout)
    f.write("\n--- STDERR ---\n")
    f.write(result.stderr)
