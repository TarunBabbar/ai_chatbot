from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load LLM
llm = OllamaLLM(model="mistral", temperature=0.5)

# Convert history to LangChain messages
def convert_gradio_history(gradio_history):
    messages = [SystemMessage(content="You are an expert software engineer. Generate clean, well-documented code based on user requirements.")]
    for msg in gradio_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

# Stream response
def generate_response(messages):
    for chunk in llm.stream(messages):
        yield chunk

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = convert_gradio_history(data["history"])
    full_response = ""
    for chunk in generate_response(messages):
        full_response += chunk
    return jsonify({"content": full_response})

# CLI loop
def chat_loop():
    history = []
    print("🧠 CodeGen Agent Ready. Type your prompt:")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append({"role": "user", "content": user_input})
        messages = convert_gradio_history(history)
        print("Agent:", end=" ", flush=True)
        full_response = ""
        for chunk in generate_response(messages):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
        history.append({"role": "assistant", "content": full_response})

# Entry point
if __name__ == "__main__":
    app.run(port=5000)