from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

system_prompt = langfuse.get_prompt(
    name='rag_app_system_prompt',
    label = 'baseline',
    type='text'
)

print(system_prompt.version)
print(system_prompt.prompt)
print(system_prompt.config)
print(system_prompt.labels)