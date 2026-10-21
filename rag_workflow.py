from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import START,END,StateGraph
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loading APIs
load_dotenv()

# Selecting llm and embedder
llm = ChatOpenAI(model='gpt-5-mini')
embedder = OpenAIEmbeddings(model='text-embedding-3-small')

# Loading documents
DIR_PATH = Path("./cleaned_transcripts")
loader = DirectoryLoader(path=DIR_PATH.as_posix(),
                         loader_cls=TextLoader,
                         show_progress=True)
documents = loader.load()

# Splitting documnets
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)
chunks = splitter.split_documents(documents=documents)

# Cretaing vector-store and persisting the chunks-embeddings
PERSIST_DIR_PATH=Path('./saved-embeddings')
vector_store = Chroma.from_documents(documents=chunks,
                                     embedding=embedder,
                                     persist_directory=PERSIST_DIR_PATH.as_posix())

# building retriever
retriever = vector_store.as_retriever(search_type='similarity',
                                      search_kwargs={'k':3})

# Defining State

class RagState(TypedDict):

    query : str
    retrived_docs : list[Document]
    context : str
    prompt : ChatPromptTemplate
    response : str

def retrieve(state : RagState)->dict:

    query = state['query']
    retrived_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrived_docs])

    return {'retrived_docs':retrived_docs,'context':context}

def augmentation(state:RagState)->dict:

    prompt = ChatPromptTemplate.from_messages([
        ('system',"""You are a helpful assistant. Answer the user query
                      based on the given context only. If you do not know the answer
                      say I don't know. Do not add any preamble to the response"""),
        ('human',"context:{context}\n\nquery:{query}")
    ])

    return {'prompt' : prompt}

def generation(state:RagState)->dict:

    query = state['query']
    context = state['context']
    prompt = state['prompt']

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'query':query,'context':context})

    return {'response' : response}

# create the graph

graph_builder = StateGraph(RagState)

graph_builder.add_node('retrieve',retrieve)
graph_builder.add_node('augmentation',augmentation)
graph_builder.add_node('generation',generation)

graph_builder.add_edge(START,'retrieve')
graph_builder.add_edge('retrieve','augmentation')
graph_builder.add_edge('augmentation',END)

graph = graph_builder.compile()