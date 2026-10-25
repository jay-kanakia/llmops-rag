import numpy as np
import dagshub
import mlflow
import json

from utils.mlflow_utils import get_metrics_from_run
from pathlib import Path

ROOT_DIR = Path(__file__).parent
THRESHOLD_PATH = ROOT_DIR/"threshold.json"

def calculate_noise_threshold(metrics:list[dict[str,float]])->dict:

    all_metric = []

    for metric in metrics:
        metric_names = list(metric.keys())
        metric_scores = list(metric.values())
        all_metric.append(metric_scores)

    matrix = np.array(all_metric)
    stds = np.std(matrix,axis=0,ddof=1).tolist()

    return {key:val for key,val in zip(metric_names,stds)}

def save_threshold(threshold_path : Path | str,stds)->None:

    if threshold_path.exists():
        with open(threshold_path,"r") as file:
            threshold = json.load(file)
            noise_threshold = threshold.get("noise_threshold")
            historical_threshold = threshold.get("historical_threshold")
    
    # new threshold
    new_threshold = {}
    new_threshold['noise_threshold'] = noise_threshold
    new_threshold['historical_threshold'] = stds

    # save threshold
    with open(threshold_path,"w") as file:
        json.dump(new_threshold,file,indent=4)


if __name__ == "__main__":

    # dagshub init
    dagshub.init(repo_owner='jay-kanakia', repo_name='llmops-rag', mlflow=True)

    # set tracking uri
    mlflow.set_tracking_uri("https://dagshub.com/jay-kanakia/llmops-rag.mlflow/")

    # set the experiment name
    mlflow.set_experiment(experiment_name="rag_app")
    experiment_id = mlflow.get_experiment_by_name(name="rag_app").experiment_id

    metric_list = get_metrics_from_run(experiment_id=experiment_id,tag_name='historical_threshold')
    stds = calculate_noise_threshold(metrics=metric_list)
    save_threshold(THRESHOLD_PATH,stds)