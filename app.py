from smolagents import CodeAgent, HfApiModel, load_tool, tool
import datetime
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI
from transformers import pipeline  # Import the summarization pipeline
import spacy

from tools.web_search import DuckDuckGoSearchTool
search_tool = DuckDuckGoSearchTool()

from tools.music_tools import MusicTool
music_tool = MusicTool()

@tool
def diagnose_disease(symptoms: str) -> str:
    """Fetches health articles based on symptoms and extracts possible diseases.
    
    Args:
        symptoms: A comma-separated list of symptoms (e.g., 'fever, headache, sore throat').
    
    Returns:
        A string containing a summary of the topic or an error message.
    """
    try:
        search_tool = DuckDuckGoSearchTool()
        query = f"Possible diseases for symptoms: {symptoms}"
        search_results = search_tool.forward(query)

        if not search_results:
            return "No relevant medical information found. Please consult a doctor."

        articles = search_results.split("\n\n")[:10]  # Get top 10 articles
        possible_diseases = set()
        
        for article in articles:
            doc = nlp(article)
            for ent in doc.ents:
                if ent.label_ == "DISEASE":
                    possible_diseases.add(ent.text)

        if not possible_diseases:
            return "No specific diseases could be identified. Consider seeking medical advice."

        return f"Based on your symptoms, possible conditions could be: {', '.join(possible_diseases)}."

    except Exception as e:
        return f"An error occurred: {e}"
    
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

final_answer = FinalAnswerTool()
model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/',  # it is possible that this model may be overloaded
    custom_role_conversions=None,
)

# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        convert_temp,
        get_current_time_in_timezone,
        diagnose_disease,
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

GradioUI(agent).launch()
