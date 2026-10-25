from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from src.app.rag_workflow import graph
from src.config.parameters_config import params_config
import logging

load_dotenv()

logger = logging.getLogger(name="Evaluation dataset logger")
logger.setLevel(logging.INFO)
console_logger = logging.StreamHandler()
console_logger.setLevel(logging.INFO)
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
logger.addHandler(console_logger)


def generate_evaluation_dataset():

    ROOT_DIR = Path(__file__).parent.parent.parent
    GOLDEN_DATASET_PATH = ROOT_DIR/"data"/"evaluation"/"goldens"/"golden_dataset_final.json"
    EVAL_DATASET_DIR = ROOT_DIR/"data"/"evaluation"/"eval_dataset"

    EVAL_DATASET_DIR.mkdir(exist_ok=True,parents=True)

    # dataset to read goldens from
    golden_dataset = EvaluationDataset()
    golden_dataset.add_goldens_from_json_file(file_path=GOLDEN_DATASET_PATH.as_posix())
    #print(len(golden_dataset.goldens))

    # dataset to hold produced test case
    eval_dataset = EvaluationDataset()

    for count, golden in enumerate(golden_dataset.goldens[:7],1):
        result = graph.invoke({'query':golden.input})
        #print(result['response'])
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=result.get("response"),
            expected_output=golden.expected_output,
            retrieval_context= [doc.page_content for doc in result.get("retrived_docs")]
        )
        eval_dataset.add_test_case(test_case=test_case)
        logger.info(msg=f"Added test case no. {count}")


    eval_dataset.save_as(
            file_type='json',
            directory=EVAL_DATASET_DIR,
            file_name=params_config.evaluation_dataset.evaluation_dataset_file_name,
            include_test_cases=True
    )

if __name__ == "__main__":
    generate_evaluation_dataset()