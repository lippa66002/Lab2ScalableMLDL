import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Configuration for available models
MODELS = {
    "Llama 3.2 1B (Code Docs)": {
        "repo_id": "lippa6602/llama-3.2-1b-code-documentation-Q4_K_M-GGUF",
        "filename": "llama-3.2-1b-code-documentation-q4_k_m.gguf",
        "chat_format": "llama-3"
    },
    "Llama 3.2 1B (Finetome)": {
        "repo_id": "lippa6602/llama-3.2-1b-finetome-optimized-Q4_K_M-GGUF",
        "filename": "llama-3.2-1b-finetome-optimized-q4_k_m.gguf",
        "chat_format": "llama-3"
    },
    "Qwen 0.5B (Code Docs)": {
        "repo_id": "lippa6602/qwen-0.5b-code-documentation-Q4_K_M-GGUF",
        "filename": "qwen-0.5b-code-documentation-q4_k_m.gguf",
        "chat_format": "chatml"
    },
    "Qwen 0.5B (Finetome)": {
        "repo_id": "lippa6602/qwen-0.5b-finetome-Q4_K_M-GGUF",
        "filename": "qwen-0.5b-finetome-q4_k_m.gguf",
        "chat_format": "chatml"
    }
}

# Global variables to track state
current_llm = None
current_model_name = None

def get_model(model_name):
    """
    Returns the requested model instance. 
    Reloads only if the requested model is different from the currently loaded one.
    """
    global current_llm, current_model_name
    
    if current_llm is not None and current_model_name == model_name:
        return current_llm

    print(f"Loading model: {model_name}...")
    config = MODELS[model_name]
    
    model_path = hf_hub_download(
        repo_id=config["repo_id"], 
        filename=config["filename"]
    )
    
    # Unload previous model from memory if it exists
    if current_llm is not None:
        del current_llm
    
    current_llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
        chat_format=config.get("chat_format")
    )
    current_model_name = model_name
    print(f"Model {model_name} loaded successfully.")
    
    return current_llm

def user_message(message, history):
    """
    Updates the history with the user's message immediately.
    """
    if history is None:
        history = []
    return "", history + [{"role": "user", "content": message}]

def bot_response(
    history: list[dict],
    system_message,
    max_tokens,
    temperature,
    top_p,
    model_name,
    code_context
):
    """
    Generates a streaming response using the selected Llama model
    and the provided code context.
    """
    # Ensure the correct model is loaded
    llm = get_model(model_name)
    
    if not history:
        return

    # Extract the last message (current user prompt) and previous history
    last_user_message = history[-1]["content"]
    previous_history = history[:-1]

    # Inject code context into the system message
    full_system_message = system_message
    if code_context and code_context.strip():
        full_system_message += f"\n\n### Context Code:\n```\n{code_context}\n```"

    messages = [{"role": "system", "content": full_system_message}]
    messages.extend(previous_history)
    messages.append({"role": "user", "content": last_user_message})

    # Initialize assistant message in history
    history.append({"role": "assistant", "content": ""})

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        stream=True
    )

    response_text = ""
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            response_text += token
            # Update the last message in history with the accumulated response
            history[-1]["content"] = response_text
            yield history

# CSS to enforce equal height on the Code component to match the Chatbot
CUSTOM_CSS = """
#code_context {
    height: 500px !important;
}
"""

with gr.Blocks(fill_height=True, css=CUSTOM_CSS) as demo:
    # 1. Top Section: Intro
    gr.Markdown(
        """
        # 🤖 Small LLM Coding Assistant
        Welcome! This tool allows you to chat with small, efficient language models specialized in coding.
        
        **Available Models:**
        * **Llama 3.2 1B**: A robust 1 billion parameter model from Meta.
        * **Qwen 2.5 0.5B**: A highly efficient 0.5 billion parameter model from Alibaba Cloud.
        
        Both models are available in two flavors:
        1. **Finetome**: General instruction tuning.
        2. **Code Docs**: Specialized training on code documentation.
        
        Paste your code in the **Context Code** editor on the right to give the model context, and ask your questions on the left.
        """
    )

    # 2. Middle Section: Split View (Chat | Code)
    with gr.Row(equal_height=True):
        # LEFT COLUMN: Chat Interface
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Chat",
                type="messages",
                height=500
            )
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask a question about the code...",
                lines=1
            )

        # RIGHT COLUMN: Code Context
        with gr.Column(scale=1):
            code_input = gr.Code(
                label="Context Code",
                language="python", 
                lines=20,
                interactive=True,
                elem_id="code_context"
            )

    # 3. Bottom Section: Settings & Explanations
    with gr.Accordion("⚙️ Model Settings & Parameters", open=True):
        gr.Markdown(
            """
            ### Settings Guide
            * **Model Selection**: Choose between Llama (1B) and Qwen (0.5B), and select the training dataset (Finetome for general instructions, Code Docs for documentation tasks).
            * **Max new tokens**: Limits the length of the response. Increase if answers are cut off.
            * **Temperature**: Controls creativity. Lower (`0.1`) is precise/deterministic; Higher (`0.9`) is creative/varied.
            * **Top-p**: Controls vocabulary diversity. Lower values restrict the model to likely words.
            * **System Message**: Defines the persona and behavior of the AI (e.g., "You are a helpful coding assistant").
            """
        )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()), 
                value="Llama 3.2 1B (Code Docs)", 
                label="Model Selection",
                interactive=True
            )
            max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.1, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
        
        system_msg = gr.Textbox(
            value="You are a coding assistant. Use the provided code context to answer questions. Keep your answers short, instructive and to the points.", 
            label="System message",
            lines=1
        )

    # Event Wiring
    # 1. User submits message -> updates chatbot history
    msg_submit = msg.submit(
        user_message, 
        [msg, chatbot], 
        [msg, chatbot], 
        queue=False
    ).then(
        # 2. Bot responds -> streams updates to chatbot history
        bot_response,
        [chatbot, system_msg, max_tokens, temperature, top_p, model_dropdown, code_input],
        [chatbot]
    )
    

if __name__ == "__main__":
    # Initialize the default model on startup
    get_model("Llama 3.2 1B (Code Docs)")
    demo.launch()