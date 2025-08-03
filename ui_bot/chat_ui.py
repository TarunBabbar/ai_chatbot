import gradio as gr
import time
from ui_bot.llama_chat import generate_response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def stream_response(message, history):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message)
    ]   

    # Generate bot response
    try:
        stream = generate_response(messages)
    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]
        return 
    
    # Stream chunk-by-chunk
    partial = ""
    for chunk in stream:
        partial += chunk
        yield [
            {"role": "assistant", "content": partial.strip()}
        ]
        time.sleep(0.05)  # Optional: simulate typing effect

# Launch the Gradio chat interface
gr.ChatInterface(
    fn=stream_response,
    title="TarunBot",
    chatbot=gr.Chatbot(type="messages"),
    type="messages"
).launch()
