import os
import base64
from typing import List
from langchain.schema import HumanMessage
from typing import TypedDict, Annotated, List, Any, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate  # Add this import
import requests
from smolagents import HfApiModel, LiteLLMModel


class AgentState(TypedDict):
    # The input document
    input_file: Optional[str]  # Contains file path, type (PNG)
    messages: Annotated[list[AnyMessage], add_messages]


def select_model():
    # Get model name and endpoint from environment variables
    # Make sure to explicitly strip any whitespace that might be in the environment variable
    # ollama_model = os.getenv("OLLAMA_MODEL", "hf.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:latest").strip()
    ollama_model = os.getenv("OLLAMA_MODEL", "qwq:latest").strip()
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").strip()
    
    # Ensure we have a valid model name
    if not ollama_model or ollama_model.startswith("ollama/"):
        # Extract the actual model name if it starts with "ollama/"
        if ollama_model.startswith("ollama/"):
            ollama_model = ollama_model[len("ollama/"):].strip()
        else:
            # Fallback to a known model if none is specified
            ollama_model = "llama2"
    
    # Initialize the Ollama LLM with explicit model parameter
    llm = Ollama(
        model=ollama_model,
        base_url=ollama_endpoint,
        request_timeout=60.0  # Increase timeout for large models
    )
    return llm

# choose API model or ollama model
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




def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
    """
    
    try:
        # Read the image file and encode it in base64
        with open(img_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Get model name and endpoint from environment variables
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:27b").strip()
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").strip()
        
        # Format for Ollama's API
        api_url = f"{ollama_endpoint}/api/chat"
        
        # Create payload for Ollama chat API
        payload = {
            "model": ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract all text from this image. Return only the text, no explanations.",
                    "images": [image_base64]
                }
            ],
            "stream": False
        }
        
        # Make the API request
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        extracted_text = result.get("message", {}).get("content", "").strip()
        
        return extracted_text
    
    except Exception as e:
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

# You can remove or simplify the vision_model function since we're
# making direct API calls to Ollama
# def vision_model():
#     # This function is kept as a placeholder for compatibility
#     # But it's not used in the direct API approach
#     return None



    

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b


def assistant(state: AgentState):
    # Prepare a system message describing the available tools.
    textual_description_of_tool = """
    extract_text(img_path: str) -> str:
        Extract text from an image file using a multimodal model.

        Args:
            img_path: A local image file path (string).

        Returns:
            A single string containing the concatenated text extracted from the image.
    divide(a: int, b: int) -> float:
        Divide a and b.
    """
    image = state["input_file"]
    sys_msg = SystemMessage(
        content=f"You are a helpful agent that can analyze images and perform computations using the provided tools:\n{textual_description_of_tool}\nCurrently loaded image: {image}"
    )

    tools = [divide, extract_text]
    # llm = select_model()
    llm = choose_model()

    # Attempt to bind tools if possible; otherwise, use the base model.
    if hasattr(llm, "bind_tools"):
        try:
            llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
        except NotImplementedError:
            llm_with_tools = llm
    else:
        llm_with_tools = llm

    messages_input = [sys_msg] + state["messages"]

    # Try various invocation methods.
    try:
        if hasattr(llm_with_tools, "invoke"):
            response = llm_with_tools.invoke(messages_input)
        elif hasattr(llm_with_tools, "predict_messages"):
            response = llm_with_tools.predict_messages(messages_input)
        elif hasattr(llm_with_tools, "predict"):
            # Fallback: convert messages to a single prompt string.
            prompt_text = "\n".join(
                [msg.content if hasattr(msg, "content") else str(msg) for msg in messages_input]
            )
            # Create a proper PromptTemplate object
            prompt = PromptTemplate(template=prompt_text)
            response_text = llm_with_tools.predict(prompt)
            response = SystemMessage(content=response_text)
        else:
            raise AttributeError("Model does not have a valid method to process messages.")
    except Exception as e:
        print(f"Error in assistant: {str(e)}")
        response = SystemMessage(content=f"I encountered an error while processing your request: {str(e)}")

    return {"messages": [response], "input_file": state["input_file"]}







def graph_show():
    # Define tools
    tools = [divide, extract_text]
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    react_graph = builder.compile()

    # (Optional) Save or display the graph image
    # png_data = react_graph.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)
    # print("Graph saved as graph.png")

    # Test the division capability
    # messages = [HumanMessage(content="Divide 6790 by 5")]
    # result = react_graph.invoke({"messages": messages, "input_file": None})
    
   

    messages = [HumanMessage(content="According the note provided by MR wayne in the provided images. What's the list of items I should buy for the dinner menu ?")]
    result = react_graph.invoke({"messages": messages,"input_file":"Batman_training_and_meals.png"})

    for m in result['messages']:
        try:
            m.pretty_print()
        except AttributeError:
            print(m.content)







if __name__ == '__main__':
    graph_show()


    # path = "Batman_training_and_meals.png"
    # print(extract_text(path))
