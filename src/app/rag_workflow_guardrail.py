import asyncio
from typing import TypedDict,Literal
from langgraph.graph import StateGraph,START,END
#from src.config.parameters_config import params_config
from langfuse import get_client
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails.errors import ValidationError

from src.app.clients import llm,app_config,logger
from src.app.guardrail import validate_input,validate_retrieval,validate_output

langfuse = get_client()
system_prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=app_config.prompt_label
)

CRITICAL_FALLBACK_MESSAGE = (
    "Something went wrong while processing your request and it could not be "
    "completed safely. Please try again in a moment."
)
SOFT_FALLBACK_MESSAGE = (
    "I couldn't produce a fully reliable answer to that question. Could you "
    "rephrase it or ask something more specific?"
)

class GuardrailRAGState(TypedDict):

    query : str
    retrieved_docs : list[Document]
    context : str
    prompt : ChatPromptTemplate
    response : str

    guardrail_status : Literal['ok','refrain','exception']
    guardrail_stage : Literal['input','retrieval','output']
    guardrail_message : str


def retrieve(state : GuardrailRAGState)->dict:

    query = state['query']
    retrived_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrived_docs])

    return {'retrived_docs':retrived_docs,'context':context}

def augmentation(state:GuardrailRAGState)->dict:

    prompt = ChatPromptTemplate.from_messages([
        ('system',system_prompt.prompt),
        ('human',"context:{context}\n\nquery:{query}")
    ])

    return {'prompt' : prompt}

def generation(state:GuardrailRAGState)->dict:

    query = state['query']
    context = state['context']
    prompt = state['prompt']

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'query':query,'context':context})

    return {'response' : response}

def input_guardrail_node(state:GuardrailRAGState)->dict:
    query = state['query']
    try:
       outcome = asyncio.run(validate_input(query=query))
    except ValidationError as e:
        logger.error(f"Input Guardrail error {e}")
        return{
            'guardrail_status' : 'exception',
            'guardrail_stage' : 'input',
            'guardrail_message' : str(e)
        }

    if not outcome.validated_output:
        logger.warning(f"Input Guardrail warning : refrain {outcome.error}")
        return{
            'guardrail_status' : 'refrain',
            'guardrail_stage' : 'input',
            'guardrail_message' : outcome.error or "Input failed validation"
        }

    # GuardrailsPII uses on_fail="fix" -- validated_output carries the
    # PII-redacted query. Forward it so every downstream node retrieves
    # and generates against the redacted text, not the raw original.
    return {
        "query": outcome.validated_output or query,
        "guardrail_status": "ok",
        "guardrail_stage": "input",
        "guardrail_message": "",
    }

def retrieval_guardrail_node(state:GuardrailRAGState)->dict:
    context  = state['context']

    try:
        outcome = asyncio.run(validate_retrieval(context=context))
    except ValidationError as e:
        logger.error(f"Retrieval Guardrail error {e}")
        return{
            'guardrail_status' : 'exception',
            'guardrail_stage' : 'retrieval',
            'guardrail_message' : str(e)
        }

    if not outcome.validated_output:
        logger.warning(f"Retrieval Guardrail warning : refrain {outcome.error}")
        return{
            'guardrail_status' : 'refrain',
            'guardrail_stage' : 'retrieval',
            'guardrail_message' : outcome.error or "Retrieval failed validation"
        }
    
    # GuardrailsPII uses on_fail="fix" -- validated_output carries the
    # PII-redacted query. Forward it so every downstream node retrieves
    # and generates against the redacted text, not the raw original.
    return {
        "guardrail_status": "ok",
        "guardrail_stage": "retrieval",
        "guardrail_message": "",
    }


def output_guardrail_node(state:GuardrailRAGState)->dict:
    query = state['query']
    context = state['context']
    response = state['response']
    
    try:
        outcome = asyncio.run(validate_output(query=query,context=context,response=response))
    except ValidationError as e:
        logger.error(f"Output Guardrail error {e}")
        return{
            'guardrail_status' : 'exception',
            'guardrail_stage' : 'output',
            'guardrail_message' : str(e)
        }
    
    if not outcome.validated_output:
        logger.warning(f"Output Guardrail warning : refrain {outcome.error}")
        return{
            'guardrail_status' : 'refrain',
            'guardrail_stage' : 'output',
            'guardrail_message' : outcome.error or "Output failed validation"
        }
        
    # GuardrailsPII uses on_fail="fix" -- validated_output carries the
    # PII-redacted query. Forward it so every downstream node retrieves
    # and generates against the redacted text, not the raw original.
    return {
        "guardrail_status": "ok",
        "guardrail_stage": "output",
        "guardrail_message": "",
    }


def guardrail_router(state:GuardrailRAGState)->str:
    return state['guardrail_status']

def error_fallback_node(state:GuardrailRAGState)->dict:
    logger.error(
        f"Error stage : {state.get('guardrail_stage')}"
        f"Error message : {state.get('guardrail_message')}"
    )
    return {
        'response' : CRITICAL_FALLBACK_MESSAGE,
        'retrieved_docs' : state.get('retrieved_docs',[])
    }


def soft_fallback_node(state:GuardrailRAGState)->dict:
    logger.warning(
        f"Warning stage : {state.get('guardrail_stage')}"
        f"Warning message : {state.get('guardrail_message')}"
    )

    return {
        'response' : SOFT_FALLBACK_MESSAGE,
        'retrieved_docs' : state.get('retrieved_docs',[])
    }

# cerate the graph
graph_builder = StateGraph(GuardrailRAGState)
graph_builder.add_node('input_guardrail_node',input_guardrail_node)
graph_builder.add_node('retrieve',retrieve)
graph_builder.add_node('retrieval_guardrail_node',retrieval_guardrail_node)
graph_builder.add_node('augmentation',augmentation)
graph_builder.add_node('generation',generation)
graph_builder.add_node('output_guardrail_node',output_guardrail_node)

graph_builder.add_node('error_fallback_node',error_fallback_node)
graph_builder.add_node('soft_fallback_node',soft_fallback_node)

graph_builder.add_edge(START,'input_guardrail_node')
graph_builder.add_conditional_edges('input_guardrail_node',guardrail_router,
                                    {'ok':'retrieve','exception':'error_fallback_node','refrain':'soft_fallback_node'})
graph_builder.add_edge('retrieve',retrieval_guardrail_node)
graph_builder.add_conditional_edges('retrieval_guardrail_node',guardrail_router,
                                    {'ok':'augmentation','exception':'error_fallback_node','refrain':'soft_fallback_node'})
graph_builder.add_edge('augmentation','generation')
graph_builder.add_conditional_edges('generation',guardrail_router,
                                     {'ok':END,'exception':'error_fallback_node','refrain':'soft_fallback_node'})
graph_builder.add_edge('error_fallback_node',END)
graph_builder.add_edge('soft_fallback_node',END)

graph = graph_builder.compile()