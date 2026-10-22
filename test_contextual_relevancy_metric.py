from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()

rag_input =  "How does the K-Nearest Neighbors (KNN) algorithm work?"

retrieval_context = [
    "KNN is a non-parametric algorithm that classifies a data point based on the majority class among its K nearest neighbors in the feature space. It requires no training phase, since the entire dataset is stored and used at prediction time. This makes prediction slow for large datasets, since distance must be computed to every point.",
    "Choosing the right value of K is important — small K makes the model sensitive to noise, while large K smooths the decision boundary but may include points from other classes.",
    "Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. KNN is often compared to SVMs as both are used for classification, but SVMs are parametric and build an explicit decision boundary during training, unlike KNN's lazy learning approach."
    ]

test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context
)

contextual_relevancy_metric = ContextualRelevancyMetric(verbose_mode=True)

score = contextual_relevancy_metric.measure(test_case=test_case)
score