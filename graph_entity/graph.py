import os
import uuid
from typing import Literal
from langchain_ollama import ChatOllama
from utils.api_base_url import ApiConfig

from contextlib import AsyncExitStack
from retriever.retrieve import vector_store
from dotenv import load_dotenv, find_dotenv
from agentics.agentic import AgriAssistAgentic
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, TypedDict, List, Any, Optional

 # llama3 gemma12B_v gemma4B_v qwen3 gemini deepseek qwen3-coder
 
# load_dotenv(override=True)
load_dotenv(find_dotenv()) # read local .env file or other file through find_dotenv

node_map = {"stock":"stock", "disease":"disease", "princing":"princing",  "prescrition":"prescrition", "market":"market", "retrieval":"retrieval" }
response_confidence = ("medium","high")

agriAssistAgentic = AgriAssistAgentic()

class RouteQuery(TypedDict):
    """Route query to destination expert."""
    destination: Literal["stock", "disease", "princing", "market", "prescrition"]

# For LangGraph, we will define the state of the graph to hold the query, destination, and final answer.
class State(TypedDict):
    result: Annotated[List[Any], add_messages]
    query: str
    answer: str
    origin: RouteQuery
    newquery: str
    destination: RouteQuery
    confidence: Optional[str]
    recursionlimit: str = "0"
    
async def retrieval_route(state: State ):
    """ Retrieval the information in the document through RAG for each prescrition """
    
    retrieval = await agriAssistAgentic.retrieval_agent(state["query"], model_name="llama3.2")
    # retrieval = await agriAssistAgentic.chain_retrieval_agent(state["query"], model_name="llama3.2")
    
    results = state["result"]
    state["origin"] = "retrieval"
    state["answer"] = retrieval
    state["result"] = results + [retrieval]
    state["confidence"] = retrieval["confidence"]
    
    return state

async def prescrition_route(state: State):
    """ And one node for each prescrition async def prescrition_route(state: State, config: RunnableConfig) """

    if str(state["origin"]) == "prescrition":
        prescrition = await agriAssistAgentic.prescrition_agent(state["query"], model_name="qwen3")
        # prescrition = await agriAssistAgentic.chain_prescrition_agent(state["newquery"], model_name="qwen3")
    else: 
        # prescrition = await agriAssistAgentic.chain_prescrition_agent(state["query"], model_name="llama3")
        prescrition = await agriAssistAgentic.prescrition_agent(state["query"], model_name="qwen3")
        
    results = state["result"]
    state["origin"] = "prescrition"
    state["answer"] = prescrition
    state["result"] = results + [prescrition]
    state["confidence"] = prescrition["confidence"]
    
    return state

async def princing_route(state: State ):
    """ And one node for each prompt princing
        async def princing_route(state: State, config: RunnableConfig) """
    
    if str(state["origin"]) == "princing":
        # princing = await agriAssistAgentic.chain_princing_agent(state["newquery"], model_name="qwen3")
        princing = await agriAssistAgentic.princing_agent(state["query"], model_name="qwen3")
    else: 
        # princing = await agriAssistAgentic.chain_princing_agent(state["query"], model_name="llama3")
        princing = await agriAssistAgentic.princing_agent(state["query"], model_name="qwen3")

    results = state["result"]
    state["origin"] = "princing"
    state["answer"] = princing
    state["result"] = results + [princing]
    state["confidence"] = princing["confidence"]
    
    return state

async def market_route(state: State ):
    """ And one node for each prompt marketing
        async def market_route(state: State, config: RunnableConfig) """
    
    # market = await agriAssistAgentic.chain_market_agent(state["query"], model_name="llama3.2")
    market = await agriAssistAgentic.market_agent(state["query"], model_name="llama3.2")
    results = state["result"]
    state["answer"] = market
    state["confidence"] = None
    state["destination"] = "stock"
    state["result"] = results + [market]

    return state

async def stock_route(state: State ):
    """ And one node for each stock
        async def stock_route(state: State, config: RunnableConfig) """
        
    # stock = await agriAssistAgentic.chain_stock_agent(state["query"], model_name="llama3")
    stock = await agriAssistAgentic.stock_agent(state["query"], model_name="llama3")
    results = state["result"]
    state["answer"] = stock
    state["confidence"] = None
    state["recursionlimit"] = 0
    state["destination"] = "princing"
    state["result"] = results + [stock]
    
    return state

def princing_evaluation(state: State) -> Literal["princing", END]:
    """ We then define logic that selects the prompt based on the classification """
    confidence = str(state["confidence"])
    origin = str(state["origin"]).lower() 
    
    if origin is not None and origin =="princing":
        if confidence.lower() in response_confidence:
            return node_map.get("END", END)
        
    results = state["result"]
    state["result"] = results[ :-1]
    return node_map.get("princing")

def prescrition_evaluation(state: State) -> Literal["prescrition", "market"]:
    """ We then define logic that selects the prompt based on the classification """
    origin = str(state["origin"]) 
    confidence = str(state["confidence"])
    
    if origin is not None and (origin in ("prescrition","retrieval")):
        if confidence.lower() in response_confidence:
            # state["recursionlimit"] = 0
            return node_map.get("market")
        
    if int(state["recursionlimit"]) > 20:
        # state["recursionlimit"] = 0
        return node_map.get("market")
    
    results = state["result"]
    state["result"] = results[ :-1]
    return node_map.get("prescrition", "prescrition")

# One "Super-Step" of the graph represents one invocation of passing messages between agents.
# In idomatic LangGraph, you call invoke to run your graph for each super-step; for each interaction.
# The reducer handles state updates automatically within one super-step, but not between them.
# That is what checkpointing achieves

async def graph_builder():
    
    db_path = "graph_memory/graph_memory.db"
    # stack = AsyncExitStack()
    # sql_memory = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db_path))
    sql_memory = InMemorySaver()
    
    # Set up Graph Builder with State
    graph = StateGraph(State)

    # Adding nodes
    graph.add_node("stock", stock_route)
    graph.add_node("princing", princing_route)
    graph.add_node("market", market_route)
    graph.add_node("prescrition", prescrition_route)
    graph.add_node("retrieval", retrieval_route)

    # Adding edges
    graph.add_edge(START, "retrieval")
    graph.add_conditional_edges("retrieval", prescrition_evaluation)
    graph.add_conditional_edges("prescrition", prescrition_evaluation)
    
    graph.add_edge("market", "stock")
    graph.add_edge("stock", "princing")
    graph.add_conditional_edges("princing", princing_evaluation)

    # Adding sql_memory
    app = graph.compile(checkpointer=sql_memory)
    
    return app