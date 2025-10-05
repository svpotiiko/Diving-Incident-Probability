
from src.io_utils import load_config, ensure_dirs
from src.pipeline import run

if __name__ == "__main__":
    cfg = load_config("config/settings.yaml")
    ensure_dirs("outputs")
    run(cfg)
