
import os
import re
import uuid

# --- Third-party ---
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from agents import Agent, Runner
from langchain_ollama import ChatOllama
from retriever.retrieve import vector_store
from .agents_client import model_client_name_dict
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from mcp_server.mcp_server import Agentic_MCP_Server
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.chains.combine_documents import create_stuff_documents_chain
from .out_puts import StockAnalysisResult, PriceAnalysisResult, PrescritionAnalysisResult, MarketAnalysisResult
from .instructions import stock_analysis, email_instructions, pricing_analysis, market_analysis, prescrition_analysis

load_dotenv(override=True)

class AgriAssistAgentic:
    
    def __init__(self, model_name: str="llama3.2"):
        self.model_name = model_name
        self.agentic_mcp_server = None
        
    async def connect_to_servers(self):
        self.agentic_mcp_server = Agentic_MCP_Server()
        await self.agentic_mcp_server.connect_to_servers()
        return self.agentic_mcp_server
    
    async def prescrition_agent(self, crop_product:str, model_name:str) -> dict:
        
        messages = [{"role": "user", "content": " You are a educational analysis of disease management practices expert."}]
        instruction = prescrition_analysis(crop_product = crop_product)
        
        agent =  Agent(
                    name = "Prescrition Agent",
                    instructions = instruction,
                    model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
                    output_type=PrescritionAnalysisResult,)
        
        result = await Runner.run(agent, messages)
        
        prescrition = { "diseaseIdentification": result.final_output.diseaseIdentification,
                        "preventivePractices": result.final_output.preventivePractices,
                        "managementApproaches": result.final_output.managementApproaches,
                        "contributingFactors": result.final_output.contributingFactors,
                        "confidence": result.final_output.confidence,
                        }
        return prescrition
    
    async def stock_agent(self, crop_product:str, model_name:str) -> dict:
        
        messages = [{"role": "user", "content": " You are expert in conducting a comprehensive analysis of the current stock and inventory status for a given crop."}]
        instruction = stock_analysis(crop_product = crop_product)
        
        agent =  Agent(
                    name = "Stock Agent",
                    instructions = instruction,
                    model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
                    output_type=StockAnalysisResult,)
        
        result = await Runner.run(agent, messages)
        
        stock = { "stockLevels": result.final_output.stockLevels,
                  "stockDisruptions": result.final_output.stockDisruptions,
                  "stockHealth": result.final_output.stockHealth,
                  "stockComparative": result.final_output.stockComparative,
                }
        
        return stock
        
    async def market_agent(self, crop_product:str, model_name:str) -> dict:
        
        messages = [{"role": "user", "content": " You are expert in researching and analyze the distribution channels and market reach for a given crop."}]
        instruction = market_analysis(crop_product = crop_product)
        
        agent =  Agent(
                    name = "Market Agent",
                    instructions = instruction,
                    model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
                    output_type=MarketAnalysisResult,)
        
        result = await Runner.run(agent, messages)
        
        market = { "marketDistribution": result.final_output.marketDistribution,
                  "marketGeographic": result.final_output.marketGeographic,
                  "issuesDistribution": result.final_output.issuesDistribution,
                }
        
        return market
        
    async def princing_agent(self,crop_product:str, model_name:str) -> dict:
        
        messages = [{"role": "user", "content": " You are expert in perform an in-depth analysis of the current market price for a given crop"}]
        instruction = pricing_analysis(crop_product = crop_product)
        
        agent =  Agent(
                    name = "Pricing Agent",
                    instructions = instruction,
                    model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
                    output_type=PriceAnalysisResult,)
        
        result = await Runner.run(agent, messages)
            
        market = { "priceAverage": result.final_output.priceAverage,
                  "priceFactor": result.final_output.priceFactor,
                  "priceForecast": result.final_output.priceForecast,
                  "priceComparison": result.final_output.priceComparison,
                }
    
    async def disease_agent(self, crop_product:str, model_name:str) -> dict:
        
        messages = [{"role": "user", "content": " You are expert in conducting a comprehensive analysis of the current stock and inventory status."}]
        instruction = stock_analysis(crop_product = crop_product)
        
        agent =  Agent(
                    name = "Stock Agent",
                    instructions = instruction,
                    model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
                    output_type=StockAnalysisResult,)
        
        result = await Runner.run(agent, messages)
        
        disease = {"stockLevels": result.final_output.stockLevels,
                  "stockDisruptions": result.final_output.stockDisruptions,
                  "stockHealth": result.final_output.stockHealth,
                  "stockComparative": result.final_output.stockComparative,
                  }
        
        return disease
    
    async def chain_market_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
        
        market_prompt = market_analysis(crop_product)
        market_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(MarketAnalysisResult)
        
        prompt = ChatPromptTemplate.from_messages(
            [("system", f"{market_prompt}"),("human", "{input}"),])
        
        agent = create_tool_calling_agent(market_m, prompt)
        market = AgentExecutor(agent = agent, verbose=True)
        
        return market.ainvoke(crop_product)
        
    async def chain_princing_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        princing_prompt = pricing_analysis(crop_product)
        princing_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PriceAnalysisResult)
        prompt = ChatPromptTemplate.from_messages(
            [("system", f"{princing_prompt}"),("human", "{input}"),])
        
        agent = create_tool_calling_agent(princing_m, prompt)
        princing = AgentExecutor(agent = agent,  verbose=True)
        
        return princing.ainvoke(crop_product )
        
    async def chain_prescrition_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        prescrition_prompt = prescrition_analysis(crop_product)
        prescrition_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PrescritionAnalysisResult)
        
        prompt = ChatPromptTemplate.from_messages(
            [("system", f"{prescrition_prompt}"),("human", "{input}"),])
        
        agent = create_tool_calling_agent(prescrition_m, prompt)
        prescrition = AgentExecutor(agent = agent,  verbose=True)
        
        return prescrition.ainvoke(crop_product)
            
    async def chain_stock_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        stock_prompt = stock_analysis(crop_product)
        stock_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(StockAnalysisResult)
        prompt = ChatPromptTemplate.from_messages(
            [("system", f"{stock_prompt}"),("human", "{input}"),])
        
        agent = create_tool_calling_agent(stock_m, prompt)
        stock = AgentExecutor(agent = agent,  verbose=True)
        
        return stock.ainvoke(crop_product )

    async def chain_retrieval_agent(self, crop_product, model_name:str, num_chunks:int=90):
        
        # vectorstore = vector_store(num_chunks=num_chunks)
        # retrieval_prompt = prescrition_analysis(crop_product = crop_product)
        # retrieval_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PrescritionAnalysisResult)
        # prompt = ChatPromptTemplate.from_messages(
        #     [("system", f"{retrieval_prompt}"),("human", "{context}"), ("Answer"),])
        
        # document_chain = create_stuff_documents_chain(retrieval_m, prompt)
        # retrieval = create_retrieval_chain(vectorstore, document_chain)
        
        stock_prompt = prescrition_analysis(crop_product = crop_product)
        stock_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PrescritionAnalysisResult)
        prompt = ChatPromptTemplate.from_messages(
            [("system", f"{stock_prompt}"),("human", "{input}"),])
        
        agent = create_tool_calling_agent(llm=stock_m, prompt=prompt)
        retrieval = AgentExecutor(agent = agent,  verbose=True)

        return retrieval.ainvoke(crop_product)

    async def chain_email_agent(self, report:str, model_name:str) -> dict:
        
        email_prompt = email_instructions(report=report, email_tool="email_sender")
        email_m = ChatOllama(temperature=0.0, model=model_name)
        
        email =   AgentExecutor(
            model = email_m,
            # tools=[email_sender],
            system_prompt = email_prompt,
            # response_format=ToolStrategy(StockAnalysisResult)
            )
          
        return email.ainvoke(report)
    
    async def send_email_agent(self, report, 
                               to_emails:list = [f"{os.getenv("GMAIL_TO"), os.getenv("GMAIL_USER")}"], 
                               sender_email:str = os.getenv("GMAIL_USER"), 
                               model_name:str = "llama3.2", 
                               output_type=None) -> Agent:
        
        if self.agentic_mcp_server == None:
            self.connect_to_servers()
            
        instruction = email_instructions( to_emails, sender_email, report=report, email_tool="email_sender")

        return Agent(
            name = "E-mail Agent",
            instructions = instruction,
            model = self.get_model(model_name),
            mcp_servers = self.agentic_mcp_server.mcp_servers["email_server"], 
            output_type=output_type,)
    
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str, num_chunks:int=90):
        """Retrieve information to help answer a query."""
        vectorstore = vector_store(num_chunks=num_chunks)
        retrieved_docs = vectorstore.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def get_model(self, model_name: str) -> Agent:
        return model_client_name_dict.get(model_name, model_client_name_dict["ollama"])
    
    def _config(self, thread:str = 1) -> dict:
        return {"configurable": {"thread_id": thread}}