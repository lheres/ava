import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# --- Configuration ---
MODEL_NAME = "microsoft/GODEL-v1_1-base-seq2seq"
# ---------------------

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer from cache...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


def format_history(message, history):
    """
    Formats the conversation history into the string format expected by the GODEL model.
    """
    dialog_history = ""
    if history:
        for user_msg, model_msg in history:
            dialog_history += f"User: {user_msg} | Person: {model_msg} | "
    dialog_history += f"User: {message} |"
    return dialog_history


def predict(message, history):
    """
    The main prediction function that generates a response from the model.
    """
    if not tokenizer or not model:
        yield "Model not loaded. Please check the logs for errors."
        return

    # Prepare the input for the GODEL model
    instruction = "Instruction: given a dialog context, continue to respond user."
    knowledge = "" # This can be used for grounded generation, but we'll leave it empty.
    dialog_history = format_history(message, history)
    query = f"{instruction} [CONTEXT] {dialog_history} {knowledge}"
    
    inputs = tokenizer(query, return_tensors="pt")

    # Generate a response using beam search for stable, coherent output
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.2
    )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Stream the response back to the UI
    partial_response = ""
    for char in response_text:
        partial_response += char
        time.sleep(0.01)
        yield partial_response

# --- Gradio Web UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="GODEL Chatbot") as demo:
    gr.Markdown(
        """
        # 🤖 Conversational AI with Microsoft's GODEL
        A simple, un-optimized chatbot interface for the GODEL model.
        """
    )
    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me anything...", container=False, scale=7),
        title=None,
        description="Start the conversation below.",
        examples=[["Hello, how are you?"], ["What is the capital of France?"], ["Can you tell me a story?"]],
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear Conversation",
    )

if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860)

