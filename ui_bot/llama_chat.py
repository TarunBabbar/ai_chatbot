from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load LLM
llm = OllamaLLM(model="mistral", temperature=0.5)

# Convert Gradio message history to LangChain message objects
def convert_gradio_history(gradio_history):
    messages = [SystemMessage(content="You are a helpful assistant.")]
    for msg in gradio_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

# Main function used in chat_ui.py

def generate_response(messages):
    for chunk in llm.stream(messages):
        yield chunk

# Optional: Agent setup (not used in current UI)
def get_current_time():
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%H:%M:%S')}"

tools = [
    Tool.from_function(
        name="get_time",
        func=get_current_time,
        description="Returns the current system time"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
