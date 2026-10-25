import yaml
from pathlib import Path
from src.config.config_types import Params

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_PATH = ROOT_DIR/"params.yaml"

def load_config(params_path: Path | str)->dict:
    with open(params_path,'r') as file:
        yaml_config = yaml.safe_load(file)

    # validate the params config
    params_config = Params.model_validate(obj=yaml_config,strict=True)

    return params_config

params_config = load_config(params_path=PARAMS_PATH)