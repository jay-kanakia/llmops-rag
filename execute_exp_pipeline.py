import json
import yaml
import mlflow
import pprint as pp
import langfuse
import dagshub
import logging

from dotenv import load_dotenv
from pathlib import Path
from typing import Literal
from src.config.parameters_config import params_config
from langfuse import get_client

from src.data.generate_eval_datasets import generate_evaluation_dataset
#from src.data.generate_goldens import generate_golden_dataset
from src.evals.application_evals.evaluate_rag_app import evaluate_app
from utils.mlflow_utils import log_run_info

load_dotenv()

ROOT_DIR = Path(__file__).parent

langfuse = get_client()

def flatten_params(params_dict:dict)->dict:
    config_dict = {}
    
    for key, value in params_dict.items():
        if isinstance(value, dict):
            config_dict.update(flatten_params(value))
        else:
            if key in config_dict:
                raise ValueError(f"Duplicate key found: {key}")
            config_dict[key] = value
    
    return config_dict

def get_latest_result(dir_path:Path | str,pattern:str)->str:

    if not dir_path.is_dir():
        raise ValueError("Provided path is not a directory")

    file_name = []

    files = dir_path.glob(pattern=pattern)
    for file in files:
        file_name.append(file.stem)

    return max(file_name)

def get_metrics_from_result(result_json_path:Path | str)->dict:
    metrics = {}

    with open(result_json_path,'r') as file:
        result_json = json.load(file)

    for metric in result_json['metricsScores']:
        metric_name = metric['metric'].removesuffix('[GEval]').strip() if '[GEval]' in metric['metric'] else metric['metric'].strip() 
        sum_score = sum(metric['scores'])
        avg_score = round((sum_score/len(metric['scores'])),2)

        metrics[metric_name] = avg_score

    return metrics

def import_system_prompt(label:str)->str:
    system_prompt  = langfuse.get_prompt(
        name = "rag_app_system_prompt",
        type="text",
        label=label
    )
    return system_prompt.prompt

def get_artifact_name(artifact_type:Literal["eval_dataset","goldens"]):

    DATA_PATH = ROOT_DIR/"data"/"evaluation"

    if artifact_type == "eval_dataset":
        artifact_path = (DATA_PATH/"eval_dataset"/params_config.evaluation_dataset.evaluation_dataset_file_name).with_suffix(".json")

    if artifact_type == "goldens":
        artifact_path = (DATA_PATH/"goldens"/params_config.golden_dataset.golden_dataset_file_name).with_suffix(".json")

    return artifact_path

def return_code_files_path():
    CODE_DIR = ROOT_DIR/"src"

    rag_workflow_path = CODE_DIR/"app"/"rag_workflow.py"
    eval_dataset_path = CODE_DIR/"data"/"generate_eval_datasets.py"
    golden_dataset_path = CODE_DIR/"data"/"generate_goldens.py"
    evaluate_rag_app_path = CODE_DIR/"evals"/"application_evals"/"evaluate_rag_app.py"

    code_files = [rag_workflow_path,eval_dataset_path,golden_dataset_path,eval_dataset_path]

    return [file.as_posix() for file in code_files]

if __name__ == "__main__":

    # dagshub init
    dagshub.init(repo_owner='jay-kanakia', repo_name='llmops-rag', mlflow=True)

    # set tracking uri
    mlflow.set_tracking_uri("https://dagshub.com/jay-kanakia/llmops-rag.mlflow/")

    # set the experiment name
    mlflow.set_experiment(experiment_name="rag_app")

    logger=logging.getLogger('MLFlow Logger')
    logger.setLevel('DEBUG')

    console_handler=logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    #file_handler=logging.FileHandler('error.log')
    #file_handler.setLevel('ERROR')

    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    #file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    #logger.addHandler(file_handler)

    with mlflow.start_run() as run:

        # get all the params
        all_params = params_config.model_dump()
        params_dict = flatten_params(all_params)

        # log all the params at once
        mlflow.log_params(params_dict)
        logger.info("Parameters logged")

        # generate evaluation dataset
        generate_evaluation_dataset()

        # evaluate app 
        evaluate_app()
        logger.info('Evaluation completed')

        # get the latest files after evaluation
        RESULT_DIR = ROOT_DIR/"reports"/"evaluation_results"
        REPORT_DIR = ROOT_DIR/"reports"/"evaluation_report"

        latest_resultfile_name = get_latest_result(dir_path=RESULT_DIR,pattern="*.json")
        latest_reportfile_name = get_latest_result(dir_path=REPORT_DIR,pattern="*.md")

        # get latest file after eval pipeline
        result_file = (RESULT_DIR/latest_resultfile_name).with_suffix(".json").as_posix()
        report_file = (REPORT_DIR/latest_reportfile_name).with_suffix(".md").as_posix()

        # log artifacts
        mlflow.log_artifact(local_path=result_file,artifact_path="result")
        mlflow.log_artifact(local_path=report_file,artifact_path="report")

        logger.info("Results and reports are logged")

        # log the metric
        metric_dict = get_metrics_from_result(result_file)
        mlflow.log_metrics(metric_dict)
        logger.info("Metrics logged")

        # log the system prompt
        system_prompt = import_system_prompt(label=params_config.rag_app.prompt_label)
        logger.info("System Prompt logged")

        # log the dataset
        eval_dataset_artifact = get_artifact_name(artifact_type='eval_dataset')
        golden_dataset_artifact = get_artifact_name(artifact_type='goldens')

        mlflow.log_artifact(local_path=eval_dataset_artifact,artifact_path="eval_dataset")
        mlflow.log_artifact(local_path=golden_dataset_artifact,artifact_path="golden_dataset")

        logger.info("logged evaluation and golden datasets")

        # log the code files
        code_paths = return_code_files_path()

        for path in code_paths:
            mlflow.log_artifact(local_path=path,artifact_path="code")
        logger.info("Code files logged")

        # set tag
        mlflow.set_tag(key="phase",value="historical_threshold")

    # extract info from run
    run_id = run.info.run_id
    run_name = run.info.run_name

    log_run_info(run_id=run_id,run_name=run_name)
    logger.info("run info logged")

