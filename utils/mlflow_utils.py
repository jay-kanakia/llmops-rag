import json
import mlflow
import dagshub
from pathlib import Path

ROOT_DIR  = Path(__file__).parent.parent
HISTORICAL_RUN_PATH = ROOT_DIR/"historical_run.json"

def log_run_info(run_id : str,run_name:str):

    if HISTORICAL_RUN_PATH.exists():
        with open(HISTORICAL_RUN_PATH,"r") as file:
            historical_run = json.load(file)

    else:
        historical_run = []

    run_dict = {
        "run_id" : run_id,
        "run_name" : run_name
    }

    historical_run.append(run_dict)
    with open(HISTORICAL_RUN_PATH,"w") as file:
        json.dump(historical_run,file,indent=4)

def get_metrics_from_run(experiment_id:str,tag_name:str):
    searched_runs = mlflow.search_runs(experiment_ids=[experiment_id],
                       filter_string=f"tags.phase = '{tag_name}'",
                       output_format="list")

    all_metrics = []

    for run in searched_runs:
        metric_dict = run.data.metrics
        all_metrics.append(metric_dict)

    return all_metrics

if __name__ == "__main__":

    # dagshub init
    dagshub.init(repo_owner='jay-kanakia', repo_name='llmops-rag', mlflow=True)

    # set tracking uri
    mlflow.set_tracking_uri("https://dagshub.com/jay-kanakia/llmops-rag.mlflow/")

    # set the experiment name
    mlflow.set_experiment(experiment_name="rag_app")

    experiment_id = mlflow.get_experiment_by_name(name="rag_app").experiment_id

    searched_runs = get_metrics_from_run(experiment_id=experiment_id,tag_name="noise_threshold")
    print(searched_runs)




