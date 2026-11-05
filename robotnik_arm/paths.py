"""Project locations that do not depend on the current working directory."""
from pathlib import Path


def task_dir() -> Path:
    """Directory that contains the arm-task entry point."""
    here = Path(__file__).resolve().parent
    if (here / "robotnik_arm_task.py").is_file():
        return here
    raise FileNotFoundError(f"robotnik_arm_task.py is not beside {__file__}")


def assets_dir(start: Path | None = None) -> Path:
    """Directory with ``configs/robotnik_arm.json`` and ``models/``.

    A copy beside the project wins. Otherwise each parent is checked for a
    ``RoboEnv`` checkout, which is where this repository keeps the simulator.
    """
    start = task_dir() if start is None else Path(start)
    candidates = [start]
    candidates.extend(parent / "RoboEnv" for parent in start.parents)
    for candidate in candidates:
        config = candidate / "configs" / "robotnik_arm.json"
        if config.is_file() and (candidate / "models").is_dir():
            return candidate
    return start
