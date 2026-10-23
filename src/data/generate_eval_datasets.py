from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv
from pathlib import Path
from src.app.rag_workflow import graph

load_dotenv()

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

for golden in golden_dataset.goldens:
    result = graph.invoke({'query':golden.input})
    #print(result['response'])
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.get("response"),
        expected_output=golden.expected_output,
        retrieval_context= [doc.page_content for doc in result.get("retrived_docs")]
    )
    eval_dataset.add_test_case(test_case=test_case)


eval_dataset.save_as(
        file_type='json',
        directory=EVAL_DATASET_DIR,
        file_name="evaluation_dataset_final",
        include_test_cases=True
)