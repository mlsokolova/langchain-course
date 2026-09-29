from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
average_pace = 5 # km/h

@tool
def triple(num:float) -> float:
    """Multiply the input by 3"""
    return float(num) * 3


def get_effective_distance(distance_km:float, elevation_gain_m:float) -> float:
    """Calculate the effective distance of a run based on the distance and elevation gain"""
    return distance_km + (elevation_gain_m / 100)

@tool
def get_estimated_time(distance_km:float, elevation_gain_m:float) -> float:
    """Calculate the estimated time of a walking 
    based on the effective distance in kilometers and average walking pase 5 km/h"""
    effective_distance = get_effective_distance(distance_km, elevation_gain_m)
    return effective_distance / 5

tools = [TavilySearch(max_results=1),
         get_effective_distance, 
         get_estimated_time]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

