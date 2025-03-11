from smolagents import CodeAgent, HfApiModel, load_tool, tool, LiteLLMModel, ToolCallingAgent, GoogleSearchTool
import datetime
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
from transformers import pipeline  # Import the summarization pipeline
import os
from dotenv import load_dotenv

from tools.web_search import DuckDuckGoSearchTool
search_tool = DuckDuckGoSearchTool()
# search_tool = GoogleSearchTool() # need API key

# from tools.music_tools import MusicTool
# music_tool = MusicTool()

load_dotenv()
    
@tool
def convert_temp(temp: float, from_unit: str, to_unit: str) -> str:
    """A tool that converts temperature between Celsius and Fahrenheit.
    
    Args:
        temp: The temperature value to convert.
        from_unit: The unit to convert from ('C' or 'F').
        to_unit: The unit to convert to ('C' or 'F').
    
    Returns:
        A string indicating the converted temperature.
    """
    try:
        if from_unit.upper() == 'C' and to_unit.upper() == 'F':
            result = (temp * 9 / 5) + 32
            return f"{temp}°C is equal to {result:.1f}°F"
        elif from_unit.upper() == 'F' and to_unit.upper() == 'C':
            result = (temp - 32) * 5 / 9
            return f"{temp}°F is equal to {result:.1f}°C"
        else:
            return "Please use 'C' for Celsius or 'F' for Fahrenheit."
    except Exception as e:
        return f"Error converting temperature '{temp}': {str(e)}"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    
    Returns:
        A string indicating the current local time in the specified timezone.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}."
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


def choose_model():
    if os.getenv("OLLAMA_MODEL"):
        print("Using an Ollama model: ", os.getenv("OLLAMA_MODEL"))

        return LiteLLMModel(
            model_id=os.getenv("OLLAMA_MODEL"),
            api_base=os.getenv("OLLAMA_ENDPOINT"),
            api_key=os.getenv("OLLAMA_KEY"),
        )
    else:
        print("Using a HuggingFace model")
        return HfApiModel(
            max_tokens=2096,
            temperature=0.5,
            model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/',
            custom_role_conversions=None,
        )


model = choose_model()
final_answer = FinalAnswerTool()
# model = HfApiModel(
#     max_tokens=2096,
#     temperature=0.5,
#     model_id='http://localhost:8080',
#     custom_role_conversions=None,
# )

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# code snippets to execute the tool calls. e.g. Python interpreter

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        convert_temp,
        get_current_time_in_timezone,
        search_tool,
        image_generation_tool,
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


# Json blob to execute the tool calls.
# agent = ToolCallingAgent(
#     tools=[
#         final_answer,
#         convert_temp,
#         get_current_time_in_timezone,
#         search_tool,
#         image_generation_tool,
#     ], 
#     model=model
#     )

# agent.visualize()

GradioUI(agent).launch()


