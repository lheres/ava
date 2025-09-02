import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time

# --- 1. Load the pre-trained model and tokenizer ---
# This is done once when the script starts to be efficient.
print("Loading model and tokenizer...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # GPT-2 doesn't have a default pad token, so we add one.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model can't be loaded, as the app is useless without it.
    exit()


# --- 2. Define the AI's response logic ---
def ava_response(message, history):
    """
    Generates a response from the AI.
    'history' is a list of lists provided by Gradio, e.g., [['user_msg_1', 'bot_msg_1']]
    """
    # Combine the history and the new message to give the model context.
    context = ""
    for user_msg, bot_msg in history[-3:]: # Use the last 3 interactions for context
        context += f"User: {user_msg}\nAVA: {bot_msg}\n"
    context += f"User: {message}\nAVA:"

    # Encode the input and generate a response
    inputs = tokenizer.encode(context, return_tensors='tf', padding=True)

    # Generate text with parameters to encourage creative and coherent responses
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + 70,  # Generate up to 70 new tokens
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Decode the output and extract just the newly generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_response = response_text[len(context):].strip()

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
        title="Chat with AVA",
        description="AVA is a conversational AI based on GPT-2. Ask her anything!",
        theme="soft",
        examples=[["Hello, who are you?"], ["What is the meaning of life?"], ["Tell me a short story."]]
    )

# Launch the web server
demo.launch(server_name="0.0.0.0", server_port=7860)
