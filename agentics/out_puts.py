import os
from pydantic import BaseModel, Field

class StockAnalysisResult(BaseModel):
    stockLevels: str = Field(description="A valid JSON object with pricise and short stock levels description. ONLY the 'stockLevels' field")
    stockDisruptions: str = Field(description="A valid JSON object with pricise and short stock disruptions description. ONLY the 'stockDisruptions' field")
    stockHealth: str = Field(description="A valid JSON object with pricise and short stock health description. ONLY the 'stockHealth' field")
    stockComparative: str = Field(description="A valid JSON object with pricise and short stock comparative analysis description. ONLY the 'stockComparative' field")

class PriceAnalysisResult(BaseModel):
    confidence: str = Field(description="here detailed classification of the confidence level of response: High, Medium, or Low")
    priceAverage: str = Field(description="A valid JSON object with pricise and short current average market range price in USD Dollar. ONLY the 'stockLevels' field")
    priceFactor: str = Field(description="A valid JSON object with description of key factors driving the current price point. ONLY the 'priceFactor' field")
    priceForecast: str = Field(description="A valid JSON object with pricise and short price forecast or trend analysis. ONLY the 'priceForecast' field")
    priceComparison: str = Field(description="A valid JSON object with price comparison against main competitors or alternatives. ONLY the 'priceComparison' field")
            
class MarketAnalysisResult(BaseModel):
    marketGeographic: str = Field(description="strength in key geographic markets")
    marketDistribution: str = Field(description="current and primary distribution and sales channels used")
    issuesDistribution: str = Field(description="market comparison against main competitors or alternatives")

class PrescritionAnalysisResult(BaseModel):
    confidence: str = Field(description="here detailed classification of the confidence level of response: High, Medium, or Low")
    diseaseIdentification: str = Field(description="description of disease characteristics, common and scientific names, typical symptom patterns observed in affected crops")
    preventivePractices: str = Field(description="pathogen/pest characteristics, environmental conditions commonly associated with disease development, and typical spread patterns")
    managementApproaches: str = Field(description="standard agricultural practices including: commonly used treatments (active ingredients and typical application rates), timing considerations, integrated pest management options, and conventional corrective measures")
    contributingFactors: str = Field(description="standard prevention strategies including: rotation patterns, variety selection considerations, cultural management practices, sanitation protocols, monitoring guidelines, and early detection indicators")
