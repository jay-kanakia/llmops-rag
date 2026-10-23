from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig,ContextConstructionConfig,EvolutionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# define paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = ROOT_DIR/"data"/"processed"

# Creating configs
filter_config = FiltrationConfig(synthetic_input_quality_threshold=0.6,
                                 max_quality_retries=2,
                                 critic_model='gpt-5-mini')

context_config = ContextConstructionConfig(critic_model='gpt-5.4-mini',
                                           max_context_length=3,
                                           min_context_length=1,
                                           max_contexts_per_document=1,
                                           context_quality_threshold=0.7,
                                           max_retries=2,
                                           chunk_size=300,
                                           chunk_overlap=30)

evolution_config = EvolutionConfig(
    num_evolutions=2,
    evolutions=({
        Evolution.MULTICONTEXT: 0.1,
        Evolution.CONCRETIZING: 0.3,
        Evolution.CONSTRAINED: 0.4,
        Evolution.COMPARATIVE: 0.2
    })
)

# Creating synthesizer
synthesizer = Synthesizer(
    model='gpt-5.4-mini',
    filtration_config=filter_config,
    evolution_config=evolution_config
)

# helper function
def get_doc_path(directory_path:Path |str)->list[str]:
    
    directory_path = Path(directory_path)
    if directory_path.exists() and directory_path.is_dir():
        paths = directory_path.glob("*.txt")

    return [path.as_posix() for path in paths]

# generate goldens
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=get_doc_path(DATASET_DIR),
    context_construction_config=context_config,
    max_goldens_per_context=1
)

# saving goldens
GOLDEN_DIR = ROOT_DIR/"data"/"evaluation"/"goldens"
GOLDEN_DIR.mkdir(exist_ok=True)

synthesizer.save_as(
    file_type='json',
    directory=GOLDEN_DIR.as_posix(),
    file_name="golden_dataset_final"
)