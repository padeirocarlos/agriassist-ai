import os
import requests
from dotenv import load_dotenv
from datetime import datetime
load_dotenv(override=True)

dt = datetime.now()
SEASON_MAP = {"Rainy": "Rainy Season (Warm and moist conditions)", 
          "Dry":"Dry Season (Lower humidity and rainfall)", 
          "Transition":"Transition Seasons (As temperatures rise and rains begin)"}

current_year = dt.year

def season() -> str:
    month = dt.month
    
    if month in (3, 4, 5, 9, 10, 11):
        return SEASON_MAP.get("Rainy")
    
    if month in (12, 1, 2, 6, 7, 8):
        return SEASON_MAP.get("Dry")

def _get_location():
     # Get location coordinates from the IP address
    location = requests.get('https://ipinfo.io/json', verify=False).json()
    # Set parameters for the weather API call
    return f" region: {location['region']} and city: {location['city']} "

def get_location():
     # Get location coordinates from the IP address
    # location = requests.get('https://ipinfo.io/json', verify=False).json()
    # Set parameters for the weather API call
    return f" region: Mozambique and city: Maputo "
    
current_season = season()
location = get_location()
    
def stock_analysis(crop_product:str) -> str:
    instruction = f"""
        Conduct a comprehensive analysis of the current stock and inventory status for {crop_product}.
        Investigate recent supply chain news, production updates, and inventory levels.
        Make sure your findings are relevant to the current market landscape in {current_year} and suitable 
        for the growing conditions located in {location}.
        
        EQUIREMENTS:
            A detailed report with 4 key bullet points covering:
            - Current inventory levels and trends.
            - Recent stock supply chain disruptions or efficiencies.
            - Analyst sentiments on stock health.
            - Stock comparative analysis against main competitors.
        
        Respond in this format:
            {{"stock_levels": "here pricise and short stock levels description",
              "stock_disruptions": "here pricise and short stock disruptions description",
              "stock_health": "here pricise and short stock health description",
              "stock_comparative": "here pricise and short stock comparative analysis description",
            }}
        """
    return instruction

def pricing_analysis(crop_product:str) -> str:
    instruction = f"""
        Perform an in-depth analysis of the current market price for {crop_product} crop.
        Investigate the factors influencing its price, including demand, competition, raw material costs, and economic conditions.
        Ensure the data is timely and reflects the market situation in {current_year} and suitable for the growing conditions located in {location}.
        
        EQUIREMENTS:
            A concise summary with 4 key bullet points covering:
            - Current average market range price analysis in USD Dollar.
            - Key factors driving the current price point.
            - Price comparison against main competitors or alternatives.
            - Short-term price forecast or trend analysis.
        
        Respond in this format:
            {{"price_average": "here pricise and short current average market range price in USD Dollar",
              "price_factor": "here pricise and short description of key factors driving the current price point",
              "price_forecast": "here pricise and short price forecast or trend analysis",
              "price_comparison": "here pricise and short price comparison against main competitors or alternatives",
              "confidence": "here detailed classification of the confidence level of response: High, Medium, or Low",
            }}
        """
    return instruction


def market_analysis(crop_product:str) -> str:
    instruction = f"""
        Research and analyze the distribution channels and market reach for {crop_product}.
        Identify the primary sales channels (e.g., online, retail, direct), key logistical partners, and geographic coverage.
        Focus on recent changes and strategies employed in {current_year} and suitable for the growing conditions located in {location}.
        
        EQUIREMENTS:
            A concise summary with 4 key bullet points covering:
            - Current and primary distribution and sales channels used.
            - Coverage strength in key geographic markets.
            - Recent partnerships, expansions, or disruptions in the distribution market network.
        
        Respond in this format:
            {{"market_distribution": "here pricise and short current and primary distribution and sales channels used",
              "market_geographic": "here pricise and strength in key geographic markets",
              "issues_distribution": "here pricise and short market comparison against main competitors or alternatives",
            }}
        """
    return instruction

def prescrition_analysis(crop_product:str) -> str:
    
    instruction = f"""
        Perform an educational analysis of disease management practices for {crop_product} 
        crop based on agricultural research and established farming guidelines.
        Provide informational overview of disease identification, contributing factors, 
        and standard agricultural practices used by farmers and agronomists.
        Focus on general agricultural knowledge applicable to growing conditions located
        in {location} during {current_season}/{current_year}.
        
        EQUIREMENTS:
            An informational summary with 4 key components covering:
            - Disease identification characteristics and symptom patterns.
            - Environmental and biological factors associated with the condition.
            - Common management approaches used in modern agriculture.
            - Standard preventive practices recommended by agricultural extension services.
        
        Respond in this format:
            {{"disease_identification": "here precise description of disease characteristics, common and scientific names, typical symptom patterns observed in affected crops",
            "contributing_factors": "here detailed explanation of pathogen/pest characteristics, environmental conditions commonly associated with disease development, and typical spread patterns",
            "management_approaches": "here comprehensive overview of standard agricultural practices including: commonly used treatments (active ingredients and typical application rates), timing considerations, integrated pest management options, and conventional corrective measures",
            "preventive_practices": "here detailed description of standard prevention strategies including: rotation patterns, variety selection considerations, cultural management practices, sanitation protocols, monitoring guidelines, and early detection indicators",
            "confidence": "here detailed classification of the confidence level of response: High, Medium, or Low",
            }}
        """
    return instruction
    
def email_instructions(to_emails:str=f"[{os.getenv("GMAIL_TO"), os.getenv("GMAIL_USER")}]", from_emails:str=os.getenv("GMAIL_USER"), report:str="", email_tool:str="email_sender"):
    
    instruction = f"""You are able to send a nicely formatted HTML email including this detailed report: {report} \n. 
    
    Task:
        You should send one email using following emails from {from_emails} to {to_emails}, providing the report converted into clean, 
        well presented HTML with an appropriate subject line. Before send you must make sure to translate and send the email in portuguese of portugal.
    
    IMPORTANT:
        Make sure to use only tools '{email_tool}' provided in mcp_server to send the email"""
        
    return instruction

