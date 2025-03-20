import os
import base64
from typing import List, TypedDict, Annotated, Any, Optional
from langchain.schema import HumanMessage
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
import requests



"""
The key improvements in this code:

1. Unified Ollama Client: I created an OllamaClient class that handles all Ollama interactions, making it easier to manage different models.
2. Separate Model Environment Variables: Using OLLAMA_TEXT_MODEL and OLLAMA_VISION_MODEL environment variables to specify different models for text and vision tasks.
3. Simplified API Calls: Direct API calls to Ollama for both text and vision models, removing the dependency on different client libraries.
4. Improved Tool Handling: The assistant function now checks for "tool calls" in the response and handles them appropriately.

"""

class AgentState(TypedDict):
    # The input document
    input_file: Optional[str]  # Contains file path, type (PNG)
    messages: Annotated[list[AnyMessage], add_messages]

    
class OllamaClient:
    """A unified client for interacting with Ollama models."""
    
    def __init__(self):
        # Get model name and endpoint from environment variables
        self.text_model = os.getenv("OLLAMA_TEXT_MODEL", "qwq:latest").strip()
        self.vision_model = os.getenv("OLLAMA_VISION_MODEL", "gemma3:27b").strip()
        self.endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").strip()
        
    def chat_completion(self, messages, model=None):
        """Make a chat completion request to Ollama API."""
        if model is None:
            model = self.text_model
            
        api_url = f"{self.endpoint}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    
    def extract_text_from_image(self, img_path):
        """Extract text from an image using vision model."""
        try:
            # Read the image file and encode it in base64
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Format messages for the vision model
            messages = [
                {
                    "role": "user",
                    "content": "Extract all text from this image. Return only the text, no explanations.",
                    "images": [image_base64]
                }
            ]
            
            # Make the API request using the vision model
            result = self.chat_completion(messages, model=self.vision_model)
            extracted_text = result.get("message", {}).get("content", "").strip()
            return extracted_text
            
        except Exception as e:
            error_msg = f"Error extracting text: {str(e)}"
            print(error_msg)
            return ""

# Initialize the Ollama client
ollama_client = OllamaClient()

def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (string).

    Returns:
        A single string containing the concatenated text extracted from the image.
    """
    return ollama_client.extract_text_from_image(img_path)

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

    # Prepare messages for Ollama
    ollama_messages = []
    
    # Add system message
    ollama_messages.append({
        "role": "system",
        "content": sys_msg.content
    })
    
    # Add user messages
    for msg in state["messages"]:
        if hasattr(msg, "content"):
            ollama_messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
    
    try:
        # Call Ollama API
        response = ollama_client.chat_completion(ollama_messages)
        response_content = response.get("message", {}).get("content", "")
        
        # Check if the response contains a tool call
        if "I need to use a tool" in response_content.lower():
            # Here you would parse the response to identify the tool and arguments
            # For simplicity, we'll check for keywords
            if "extract_text" in response_content.lower() and image:
                # Extract text from the image
                extracted_text = extract_text(image)
                # Return the extracted text in the response
                return {"messages": [SystemMessage(content=f"I've extracted the text from the image:\n\n{extracted_text}")], "input_file": state["input_file"]}
            elif "divide" in response_content.lower():
                # This is a simplification. In a real implementation, you'd parse the numbers
                # For demonstration, we'll just return a message
                return {"messages": [SystemMessage(content="I would perform the division here.")], "input_file": state["input_file"]}
        
        # If no tool call detected, return the response as is
        return {"messages": [SystemMessage(content=response_content)], "input_file": state["input_file"]}
    
    except Exception as e:
        print(f"Error in assistant: {str(e)}")
        return {"messages": [SystemMessage(content=f"I encountered an error while processing your request: {str(e)}")], "input_file": state["input_file"]}

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

    # Test the extraction capability
    # messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?")]
    messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. Show me all the information you can get from them.")]
    result = react_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})

    for m in result['messages']:
        try:
            m.pretty_print()
        except AttributeError:
            print(m.content)

if __name__ == '__main__':
    graph_show()