import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.config.parameters_config import params_config
from pathlib import Path

load_dotenv()

app_config = params_config.rag_app

logger = logging.getLogger('Rag App')
logger.setLevel(logging.INFO)

console_handler  = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

ROOT_DIR = Path(__file__).resolve.parent.parent.parent

llm = ChatOpenAI(model=app_config.llm,temperature=0)
