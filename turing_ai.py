import gradio as gr
import tensorflow as tf
from transformers import TFAutoModelForConditionalGeneration, AutoTokenizer
import time

# --- 1. Load Microsoft's GODEL model and tokenizer ---
# We are using microsoft/GODEL-v1_1-base-seq2seq, a model designed for grounded conversation.
print("Loading model and tokenizer (microsoft/GODEL-v1_1-base-seq2seq)...")
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = TFAutoModelForConditionalGeneration.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model can't be loaded, as the app is useless without it.
    exit()


# --- 2. Define the AI's response logic ---
def ava_response(message, history):
    """
    Generates a response from the AI using the GODEL model's expected format.
    """
    # Build the dialogue history string. GODEL expects turns separated by '|'.
    dialogue_history = []
    for user_turn, bot_turn in history:
        dialogue_history.append(user_turn)
        dialogue_history.append(bot_turn)
    dialogue_history.append(message)
    
    dialogue_string = " | ".join(dialogue_history)

    # Format the input with the instruction required by the GODEL model
    prompt = f"Instruction: given a dialog context, continue to reply. Dialogue: {dialogue_string}"

    # Encode the input and generate a response
    inputs = tokenizer(prompt, return_tensors="tf")

    # Generate text. For seq2seq, max_length is the length of the *output*.
    outputs = model.generate(
        **inputs,
        max_length=80,  # Max length of the generated response
        no_repeat_ngram_size=3,
        num_beams=4,
        early_stopping=True
    )

    # Decode the output. The entire decoded string is the new response.
    new_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simulate a typing effect for a more engaging user experience
    response_stream = ""
    for char in new_response:
        response_stream += char
        time.sleep(0.02)
        yield response_stream

# --- 3. Create and launch the Gradio Web App ---
print("AVA: Systems online. Gradio interface is starting.")

# Using an explicit gr.Blocks() context to prevent rendering errors.
with gr.Blocks(theme="soft") as demo:
    gr.ChatInterface(
        fn=ava_response,
        title="Chat with AVA (Microsoft GODEL)",
        description="AVA is a conversational AI based on Microsoft's GODEL model. Ask her anything!",
        theme="soft",
        examples=[["What's your favorite book?"], ["Explain the theory of relativity in simple terms."], ["Who was Alan Turing?"]]
    )

# Launch the web server
demo.launch(server_name="0.0.0.0", server_port=7860)

