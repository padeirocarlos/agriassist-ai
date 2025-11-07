
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
from langchain.agents import create_agent
from .agents_client import model_client_name_dict
from mcp_server.mcp_server import Agentic_MCP_Server
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from .out_puts import StockAnalysisResult, PriceAnalysisResult, PrescritionAnalysisResult, MarketAnalysisResult
from .instructions import stock_analysis, email_instructions, pricing_analysis, market_analysis, prescrition_analysis, retrievel_prescrition_analysis

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
                        "content": " You are a educational analysis of disease management practices expert.",
                        "role": "user",
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
                  "content": " You are expert in perform an in-depth analysis of the current market price for a given crop",
                  "role": "user",
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
                  "content": " You are expert in perform an in-depth analysis of the current market price for a given crop",
                  "role": "user",
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
            
        market = { 
                  "confidence" : result.final_output.confidence,
                  "priceAverage": result.final_output.priceAverage,
                  "priceFactor": result.final_output.priceFactor,
                  "priceForecast": result.final_output.priceForecast,
                  "priceComparison": result.final_output.priceComparison,
                  "content": "You are expert in perform an in-depth analysis of the current market price for a given crop",
                  "role": "user",
                  }
        return market
    
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
    
    async def retrieval_agent(self, crop_product, model_name:str, num_chunks:int=90):
        messages = [{"role": "user", "content": " You are a educational analysis of disease management practices expert."}]
        retrieval_prompt = retrievel_prescrition_analysis(crop_product = crop_product)
        
        if self.agentic_mcp_server == None:
            await self.connect_to_servers()
        
        mcp_servers = self.agentic_mcp_server.mcp_servers["retrievel_server"]
            
        agent =  Agent(
            name = "Retrieval Agent",
            instructions = retrieval_prompt,
            model = self.get_model(self.model_name) if model_name is None else self.get_model(model_name),
            mcp_servers = mcp_servers,
            output_type=PrescritionAnalysisResult,)
        
        result = await Runner.run(agent, messages)
        
        retrieval = { "diseaseIdentification": result.final_output.diseaseIdentification,
                        "preventivePractices": result.final_output.preventivePractices,
                        "managementApproaches": result.final_output.managementApproaches,
                        "contributingFactors": result.final_output.contributingFactors,
                        "confidence": result.final_output.confidence,
                        "content": "You are a educational analysis of disease management practices expert.",
                        "role": "user",
                        }
        
        return retrieval
    
    async def chain_market_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
        
        market_prompt = market_analysis(crop_product)
        market_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(MarketAnalysisResult)
        
        market = create_agent(
            model=market_m,
            # tools=tools,
            system_prompt = market_prompt,
            response_format = ToolStrategy(MarketAnalysisResult),
            )
        
        return await market.ainvoke({"query": f"{crop_product}"})
        
    async def chain_princing_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        princing_prompt = pricing_analysis(crop_product)
        princing_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PriceAnalysisResult)
        
        princing = create_agent(
                model=princing_m,
                # tools=tools,
                system_prompt = princing_prompt,
                response_format = ToolStrategy(PriceAnalysisResult),
                )
        
        return await princing.ainvoke({"query": f"{crop_product}"})
        
    async def chain_prescrition_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        prescrition_prompt = prescrition_analysis(crop_product)
        prescrition_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PrescritionAnalysisResult)
        
        prescrition = create_agent(
            model=prescrition_m,
            # tools=tools,
            system_prompt = prescrition_prompt,
            response_format = ToolStrategy(PrescritionAnalysisResult),
            )
        
        return await prescrition.ainvoke({"query": f"{crop_product}"})
            
    async def chain_stock_agent(self, crop_product:str, model_name:str, thread:str=1) -> dict:
            
        stock_prompt = stock_analysis(crop_product)
        stock_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(StockAnalysisResult)
        
        stock = create_agent(
            model=stock_m,
            # tools=tools,
            system_prompt = stock_prompt,
            response_format = ToolStrategy(StockAnalysisResult),
            )
        
        return await stock.ainvoke({"query": f"{crop_product}"})
    
    async def chain_retrieval_agent(self, crop_product, model_name:str, num_chunks:int=90):
        
        retrieval_prompt = prescrition_analysis(crop_product = crop_product)
        # retrieval_m = ChatOllama(temperature=0.0, model=model_name).with_structured_output(PrescritionAnalysisResult)
        retrieval_m = ChatOllama(temperature=0.0, model=model_name)
        
        retrieval = create_agent(
            model=retrieval_m,
            # tools=[retrieve_context],
            system_prompt = retrieval_prompt,
            response_format = ToolStrategy(PrescritionAnalysisResult),
            )

        return await retrieval.ainvoke({"query": f"{crop_product}"})

    async def chain_email_agent(self, report:str, model_name:str) -> dict:
        
        email_prompt = email_instructions(report=report, email_tool="email_sender")
        email_m = ChatOllama(temperature=0.0, model=model_name)

        email = create_agent(
            model=email_m,
            tools=[self.retrieve_context],
            system_prompt = email_prompt,
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
    
    def get_model(self, model_name: str) -> Agent:
        return model_client_name_dict.get(model_name, model_client_name_dict["ollama"])
    
    def _config(self, thread:str = 1) -> dict:
        return {"configurable": {"thread_id": thread}}