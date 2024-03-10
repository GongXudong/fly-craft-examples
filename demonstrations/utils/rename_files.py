import sys
from pathlib import Path

PROJECT_ROOT_DIR: Path = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

def work(data_dir: Path, old_prefix: str, new_prefix: str):
    """rename the name of files in 'data_dir', 
    change its prefix from 'old_prefix' to 'new_prefix'

    Args:
        data_dir (Path): _description_
        old_prefix (str): _description_
        new_prefix (str): _description_
    """
    for file in data_dir.iterdir():
        if file.is_file() and file.name.startswith(old_prefix):
            new_name = file.name.replace(old_prefix, new_prefix, 1)
            new_file = file.rename(file.parent / new_name)
            print(f"rename {file.name} to {new_file.name}.")

if __name__ == "__main__":
    work(
        PROJECT_ROOT_DIR / "demonstrations" / "data" / "10hz_10_5_5_v2",
        old_prefix="trace",
        new_prefix="traj"
    )