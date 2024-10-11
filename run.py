import os
import sys
import subprocess


def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def start():
    subprocess.run(["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"])


def dev():
    subprocess.run(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [install|start|dev]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "install":
        install()
    elif command == "start":
        start()
    elif command == "dev":
        dev()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py [install|start|dev]")
        sys.exit(1)
