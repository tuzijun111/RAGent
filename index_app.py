from datasets import load_dataset
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.tools.google import GmailToolSpec
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream, ReActAgent
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Context, Event
from llama_index.utils.workflow import draw_all_possible_flows


import asyncio
import nest_asyncio
import os
import random
from pathlib import Path
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()



def select_model():
    # Get model name and endpoint from environment variables
    # Make sure to explicitly strip any whitespace that might be in the environment variable
    ollama_model = os.getenv("OLLAMA_MODEL", "hf.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:latest").strip()
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434").strip()
    
    # print(f"Using Ollama model: '{ollama_model}' at {ollama_endpoint}")
    
    # Ensure we have a valid model name
    if not ollama_model or ollama_model.startswith("ollama/"):
        # Extract the actual model name if it starts with "ollama/"
        if ollama_model.startswith("ollama/"):
            ollama_model = ollama_model[len("ollama/"):].strip()
            # print(f"Extracted model name: '{ollama_model}'")
        else:
            # Fallback to a known model if none is specified
            ollama_model = "llama2"
            # print(f"Using fallback model: '{ollama_model}'")
    
    # Initialize the Ollama LLM with explicit model parameter
    llm = Ollama(
        model=ollama_model,
        base_url=ollama_endpoint,
        request_timeout=60.0  # Increase timeout for large models
    )
    
    return llm
    
    
def rag_test(): 

    # Select the model
    llm = select_model()
    
    # Load dataset
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")

    # Create directory and save dataset as text files
    Path("data").mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(dataset):
        with open(Path("data") / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])

    # Read documents
    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()
    # documents = documents[:100]  # Reduce to 100 for testing
    print(f"Loaded {len(documents)} documents.")
    # print(documents[:2])

    # Use a local embedding model
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Configure settings
    Settings.embed_model = embedding_model
    Settings.llm = llm

    # Create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embedding_model,
        ]
    )

    # Process documents
    async def process_documents():
        return await pipeline.arun(documents=documents[:50])
    
    nodes = asyncio.run(process_documents())
    print(f"Processed {len(nodes)} nodes.")
    # print("Nodes in chroma_db:", nodes)

    # Set up ChromaDB vector store
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create index and add nodes
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    index.insert_nodes(nodes)
    print(f"Inserted {len(nodes)} nodes into the vector store index.")

    # Create a query engine with explicit parameters
    query_engine = index.as_query_engine(
        llm=llm,  # Explicitly pass the LLM to the query engine
        response_mode="tree_summarize",
    )

    # Run a query
    query_text = "Respond using a persona that describes author and travel experiences?"
    query_text1 = "What is the name of the someone that is interested in AI and techhnology?"
    
    # try:
    #     # Test the LLM with a simple query first to make sure it works
    #     print("Testing LLM with a simple query...")
    #     simple_test = llm.complete("Hello, are you working?")
    #     print(f"LLM test response: {simple_test}")
        
    #     # Then run the actual RAG query
    #     print("\nRunning RAG query...")
    #     response = query_engine.query(query_text)
    #     print("\nQuery Response:\n", response)
    # except Exception as e:
    #     print(f"Error during query: {e}")
    #     import traceback
    #     traceback.print_exc()

    response = query_engine.query(query_text)
    print("\nQuery Response:\n", response)
    # response1 = query_engine.query(query_text1)
    # print("\nQuery1 Response:\n", response1)

    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=response)
    print("Response Correctness: ", eval_result.passing)



    # Human-in-the-loop feedback
    # user_feedback = input("Is this response acceptable? (Y/N): ")
    # if user_feedback.lower() != "y":
    #     correction = input("Please provide your correction: ")
    #     # You could then log this feedback or use it to adjust the system
    #     print("Thank you for your feedback!")

def query_engine_init(name, description):
    # Set up ChromaDB vector store
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    llm = select_model()
    # Create index and add nodes
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model = embedding_model)
    query_engine = index.as_query_engine(llm=llm)

    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name= name,
        description= description,
        return_direct = False
    )
    return tool



def query_engine_test():
    def get_weather(location: str) -> str:
        """Useful for getting the weather for a given location."""
        print(f"Getting weather for {location}")
        return f"The weather in {location} is sunny"


    tool = FunctionTool.from_defaults(
        get_weather,
        name="my_weather_tool",
        description="Useful for getting the weather for a given location.",
    )
    # print(tool.call("New York"))

   
    tool = query_engine_init("some useful name", "some useful description")

    async def async_tool_call(tool):
        text = "Respond about research on the impact of AI on the future of work and society."
        response = await tool.acall(text)
        print(response)

    asyncio.run(async_tool_call(tool))



    tool_spec = GmailToolSpec()
    tool_spec_list = tool_spec.to_tool_list()
    # print(tool_spec_list)
    # print([(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list])

def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    return a / b

async def async_agent_test():
    
    llm = select_model()

    # # Create an agent workflow
    # agent = AgentWorkflow.from_tools_or_functions(
    #     tools_or_functions=[subtract, multiply, divide, add],
    #     llm=llm,
    #     system_prompt="You are a math agent that can add, subtract, multiply, and divide numbers using provided tools.",
    # )

    # handler = agent.run("What is (2 + 2) * 2?")
    
    # async for ev in handler.stream_events():
    #     if isinstance(ev, ToolCallResult):
    #         print("\nCalled tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
    #     elif isinstance(ev, AgentStream):  # showing the thought process
    #         print(ev.delta, end="", flush=True)

    # resp = await handler
    # print("\nFinal Response for calculation:", resp)

    # # Maintain conversation context
    # context = {}  # Using a simple dictionary to track state
    # response = await agent.run("My name is Bob.", ctx=context)
    # response = await agent.run("What was my name again?", ctx=context)
    # print("\nFinal Response in the context:", response)

    # Initialize a query engine tool
    try:
        tool = query_engine_init("personas", "descriptions for various types of personas")  # Ensure this function is defined
    except NameError:
        print("Error: query_engine_init() is not defined.")
        return

    # Create a RAG agent
    query_engine_agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[tool],
        llm=llm,
        system_prompt="You are a helpful assistant that has access to a database containing persona descriptions.",
    )

    handler = query_engine_agent.run(
        "Search the database for 'science fiction' and return some persona descriptions."
    )

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print("\nCalled tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
        elif isinstance(ev, AgentStream):  # showing the thought process
            print(ev.delta, end="", flush=True)

    resp = await handler
    print("\nFinal RAG Response:", resp)

def agent_test():
    asyncio.run(async_agent_test())



async def async_multi_agent_test():
    llm = select_model()
    calculator_agent = ReActAgent(
        name="calculator",
        description="Performs basic arithmetic operations",
        system_prompt="You are a calculator assistant. Use your tools for any math operation.",
        tools=[add, subtract],
        llm=llm,
    )

    query_engine_tool = query_engine_init("personas", "descriptions for various types of personas")
    
    query_agent = ReActAgent(
        name="info_lookup",
        description="Looks up information about XYZ",
        system_prompt="Use your tool to query a RAG system to answer information about XYZ",
        tools=[query_engine_tool],
        llm=llm,
    )


    # Create and run the workflow
    agent = AgentWorkflow(agents=[calculator_agent, query_agent], root_agent="calculator")

    # Run the system
    handler = agent.run(user_msg="Can you add 5 and 3?")

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print("")
            print("Called tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
        elif isinstance(ev, AgentStream):  # showing the thought process
            print(ev.delta, end="", flush=True)

    resp = await handler
    print("\nFinal Response:", resp)

def multi_agent_test():
    asyncio.run(async_multi_agent_test())


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")
    
class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str


# class MultiStepWorkflow(Workflow):
#     @step
#     async def step_one(self, ev: StartEvent) -> ProcessingEvent:
#         # Process initial data
#         return ProcessingEvent(intermediate_result="Step 1 complete")

#     @step
#     async def step_two(self, ev: ProcessingEvent) -> StopEvent:
#         # Use the intermediate result
#         final_result = f"Finished processing: {ev.intermediate_result}"
#         return StopEvent(result=final_result)

# class MultiStepWorkflow(Workflow):
#     @step
#     async def step_one(self, ev: StartEvent) -> ProcessingEvent | LoopEvent:
#         if random.randint(0, 1) == 0:
#             print("Bad thing happened")
#             return LoopEvent(loop_output="Back to step one.")
#         else:
#             print("Good thing happened")
#             return ProcessingEvent(intermediate_result="First step complete.")

#     @step
#     async def step_two(self, ev: ProcessingEvent | LoopEvent) -> StopEvent:
#         # Use the intermediate result
#         if isinstance(ev, ProcessingEvent):
#             # Use intermediate result only if the event is ProcessingEvent
#             final_result = f"Finished processing: {ev.intermediate_result}"
#         else:
#             # Handle LoopEvent separately
#             final_result = "Loop event occurred. No intermediate result available."

#         return StopEvent(result=final_result)

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent, ctx: Context) -> ProcessingEvent:
        # Process initial data
        await ctx.set("query", "What is the capital of France?")
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent, ctx: Context) -> StopEvent:
        # Use the intermediate result
        query = await ctx.get("query")
        print(f"Query: {query}")
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


    
async def async_workflow_test(): 
    # w = MyWorkflow(timeout=10, verbose=False)
    # w = MultiStepWorkflow(timeout=10, verbose=False)
    # result = await w.run()
    # print("\nResponse:", result)

    # print(draw_all_possible_flows(w))

    # we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
    llm = select_model()
    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="A helpful assistant that can use a tool to multiply numbers.",
        tools=[multiply], 
        llm=llm,
    )

    addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[add], 
        llm=llm,
    )

    # Create the workflow
    workflow = AgentWorkflow(
        agents=[multiply_agent, addition_agent],
        root_agent="multiply_agent"
    )

    response = await workflow.run(user_msg="Can you add 5 and 3?")
    print("\nResponse:", response)


def workflow_test():
    asyncio.run(async_workflow_test())  






if __name__ == '__main__':
    # rag_test()
    # query_engine_test()
    # agent_test()
    # multi_agent_test()
    workflow_test()