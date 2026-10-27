import json
import dagshub
import mlflow
import pytest

from pathlib import Path
from typing import Literal
from utils.mlflow_utils import get_metric_info,get_metrics_from_stage

# dagshub init
dagshub.init(repo_owner='jay-kanakia', repo_name='llmops-rag', mlflow=True)

# set tracking uri
mlflow.set_tracking_uri("https://dagshub.com/jay-kanakia/llmops-rag.mlflow/")

ROOT_DIR = Path(__file__).parent.parent
HISTORICAL_RUN_PATH = ROOT_DIR/"historical_run.json"
THRESHOLD_PATH = ROOT_DIR/"threshold.json"

TARGET_STAGE = "stagging"
MULTIPLER = 2


def get_latest_run_id(json_path : Path | str)->str:

    if isinstance(json_path,str):
        json_path = Path(json_path)

    if json_path.exists():
        with open(json_path,'r') as file:
            historical_runs = json.load(file)

        latest_run_id = historical_runs[-1]["run_id"]
        return latest_run_id
        
    else:
        raise ValueError("File does not exist")


def load_threshold(threshold_path : Path | str, threshold_type : Literal['noise_threshold','historical_threshold'])->dict:

    if isinstance(threshold_path,str):
        threshold_path = Path(threshold_path)

    if threshold_path.exists():
        with open (threshold_path,"r") as file:
            threshold = json.load(file)

        return threshold[threshold_type]
        
    else:
        raise ValueError("File does not exist")

# get experiment id
experiment_id = mlflow.get_experiment_by_name(name="rag_app").experiment_id
latest_run_id = get_latest_run_id(json_path=HISTORICAL_RUN_PATH)

noise_threshold = load_threshold(threshold_path=THRESHOLD_PATH,threshold_type="noise_threshold")
historical_threshold = load_threshold(threshold_path=THRESHOLD_PATH,threshold_type="historical_threshold")

stagging_metrics = get_metrics_from_stage(experiment_id=experiment_id,tag_name=TARGET_STAGE)
latest_metrics = get_metric_info(run_id=latest_run_id)

stagging_metrics_names = list(stagging_metrics.keys())
latest_metrics_names = list(latest_metrics.keys())
print(stagging_metrics)
print(latest_metrics)

def test_similar_names():
    if stagging_metrics_names!=latest_metrics_names:
        pytest.exit(reason="Comparison metrics are different") # for complete halt of test

@pytest.mark.parametrize(argnames='metric',argvalues=latest_metrics_names)
def test_regression_on_metrics(metric:str):
    
    stage_value = stagging_metrics[metric]
    latest_value = latest_metrics[metric]

    historical_th = MULTIPLER*historical_threshold[metric]
    noise_th = MULTIPLER*noise_threshold[metric]

    lower_bound = stage_value - (historical_th + noise_th)

    assert latest_value>lower_bound, f"Metric {metric} regressed"