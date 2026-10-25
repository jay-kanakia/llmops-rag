from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig,ContextConstructionConfig,EvolutionConfig
from deepeval.synthesizer.types import Evolution
from dotenv import load_dotenv
from pathlib import Path
from src.config.parameters_config import params_config

load_dotenv()

golden_configs  = params_config.golden_dataset

def generate_golden_dataset():

    # define paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATASET_DIR = ROOT_DIR/"data"/"processed"

    # Creating configs
    filter_config = FiltrationConfig(synthetic_input_quality_threshold=golden_configs.filtration_config.filtration_threshold,
                                    max_quality_retries=golden_configs.filtration_config.filtration_max_retries,
                                    critic_model=golden_configs.filtration_config.filtration_critic_model)

    context_config = ContextConstructionConfig(critic_model=golden_configs.context_construction_config.construction_critic_model,
                                            max_context_length=golden_configs.context_construction_config.max_context_length,
                                            min_context_length=golden_configs.context_construction_config.min_context_length,
                                            max_contexts_per_document=golden_configs.context_construction_config.max_contexts_per_document,
                                            context_quality_threshold=golden_configs.context_construction_config.context_quality_threshold,
                                            max_retries=golden_configs.context_construction_config.context_max_retries,
                                            chunk_size=golden_configs.context_construction_config.context_chunk_size,
                                            chunk_overlap=golden_configs.context_construction_config.context_chunk_overlap)

    evolution_config = EvolutionConfig(
        num_evolutions=golden_configs.evolution_config.num_evolution,
        evolutions=({
            Evolution.MULTICONTEXT: golden_configs.evolution_config.evolutions.multicontext,
            Evolution.CONCRETIZING: golden_configs.evolution_config.evolutions.concretizing,
            Evolution.CONSTRAINED: golden_configs.evolution_config.evolutions.constrained,
            Evolution.COMPARATIVE: golden_configs.evolution_config.evolutions.comparative
        })
    )

    # Creating synthesizer
    synthesizer = Synthesizer(
        model=golden_configs.dataset_model,
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
        max_goldens_per_context=golden_configs.max_goldens_per_context
    )

    # saving goldens
    GOLDEN_DIR = ROOT_DIR/"data"/"evaluation"/"goldens"
    GOLDEN_DIR.mkdir(exist_ok=True)

    synthesizer.save_as(
        file_type='json',
        directory=GOLDEN_DIR.as_posix(),
        file_name=golden_configs.golden_dataset_file_name
    )

if __name__ == "__main__":
    generate_golden_dataset()