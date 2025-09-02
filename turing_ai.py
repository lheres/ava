import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# --- Configuration ---
MODEL_NAME = "microsoft/GODEL-v1_1-base-seq2seq"
IPEX_AVAILABLE = False
# ---------------------

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer from cache...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # Put the model in evaluation mode. This is important for optimizations.
    model = model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# --- Apply Performance Optimizations (IPEX + torch.compile) ---
if model:
    try:
        import intel_extension_for_pytorch as ipex
        print("Applying Intel Extension for PyTorch (IPEX) optimizations...")
        # Optimize the model with IPEX, using BFloat16 for a free speed boost
        model = ipex.optimize(model, dtype=torch.bfloat16)
        print("Compiling model with torch.compile (this may take a moment)...")
        # Compile the model with the IPEX backend for maximum performance
        model = torch.compile(model, backend="ipex")
        IPEX_AVAILABLE = True
    except ImportError:
        print("WARNING: Intel Extension for PyTorch not found. Running in standard mode.")
    except Exception as e:
        print(f"WARNING: An error occurred during IPEX optimization: {e}")

# --- Mandatory Model Warmup ---
def warmup_model():
    if not all([model, tokenizer, IPEX_AVAILABLE]):
        print("Skipping warmup.")
        return
    print("Starting model warmup...")
    try:
        # Prepare a dummy input representative of a real query
        dummy_text = "Instruction: given a dialog context, continue to respond user. [CONTEXT] User: Hello EOS"
        inputs = tokenizer(dummy_text, return_tensors="pt")
        # Run generation a few times to allow torch.compile to optimize graph
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            for i in range(3):
                print(f"Warmup run {i+1}/3...")
                # **FIX**: Pass the attention_mask to the model during warmup
                _ = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=10
                )
        print("Warmup complete. Model is ready for inference.")
    except Exception as e:
        print(f"An error occurred during model warmup: {e}")

if model:
    warmup_model()


def format_history(message, history):
    dialog_history = ""
    if history:
        for user_msg, model_msg in history:
            dialog_history += f"User: {user_msg} EOS Person: {model_msg} EOS "
    dialog_history += f"User: {message} EOS"
    return dialog_history


def predict(message, history):
    if not tokenizer or not model:
        yield "Model not loaded. Please check the logs for errors."
        return

    instruction = "Instruction: given a dialog context, continue to respond user."
    knowledge = ""
    dialog_history = format_history(message, history)
    query = f"{instruction} [CONTEXT] {dialog_history} {knowledge}"
    
    inputs = tokenizer(query, return_tensors="pt")

    # Generate a response using BFloat16 mixed-precision for speed
    with torch.cpu.amp.autocast(enabled=IPEX_AVAILABLE, dtype=torch.bfloat16):
        # **FIX**: Pass the attention_mask to the model during prediction
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            min_length=8,
            top_p=0.9,
            do_sample=True,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    for char in response_text:
        yield char
        time.sleep(0.01)

# --- Gradio Web UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="GODEL Chatbot") as demo:
    gr.Markdown(
        """
        # 🤖 Conversational AI with Microsoft's GODEL (Optimized)
        This demo runs a highly optimized version of the GODEL model using Intel IPEX and `torch.compile`.
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

