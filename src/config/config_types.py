from pydantic import BaseModel

class RagAppConfig(BaseModel):

  prompt_label: str
  llm: str
  embedder_model: str
  embedder_dimensions: int
  chunk_size: int
  chunk_overlap: int
  tokenizer_encoding: str
  collection_name: str
  search_type: str
  k: int

  model_config = {"extra" : "forbid"}

class FiltrationConfig(BaseModel):
  filtration_threshold: float
  filtration_max_retries: int
  filtration_critic_model: str

  model_config = {"extra" : "forbid"}

class ContextConstrutionConfig(BaseModel):
  construction_critic_model: str
  max_context_length: int
  min_context_length: int
  max_contexts_per_document: int
  context_quality_threshold: float
  context_max_retries: int
  context_chunk_size: int
  context_chunk_overlap: int

  model_config = {"extra" : "forbid"}

class EvolutionConfig(BaseModel):
  num_evolution: int
  evolutions: dict[str,float]  

  model_config = {"extra" : "forbid"}

class GoldenDatasetConfig(BaseModel):
  dataset_model: str
  max_goldens_per_context: int
  golden_dataset_file_name: str

  filtration_config: FiltrationConfig
  context_construction_config: ContextConstrutionConfig
  evolution_config: EvolutionConfig

  model_config = {"extra" : "forbid"}

class EvaluationDatasetConfig(BaseModel):
  evaluation_dataset_file_name: str

  model_config = {"extra" : "forbid"}

class AsyncConfig(BaseModel):
  throttle_value: int
  max_concurrent: int

  model_config = {"extra" : "forbid"}

class DisplayConfig(BaseModel):
  result_dir: str
  report_dir: str

  model_config = {"extra" : "forbid"}

class EvaluationConfig(BaseModel):
  judge_llm: str
  async_config: AsyncConfig
  display_config: DisplayConfig

  model_config = {"extra" : "forbid"}

class Params(BaseModel):
  rag_app: RagAppConfig
  golden_dataset: GoldenDatasetConfig
  evaluation_dataset: EvaluationDatasetConfig
  evaluation: EvaluationConfig

  model_config = {"extra" : "forbid"}

