import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time
import random
import gradio as gr

# --- AI Initialization (same as before) ---
print("AVA: Booting up and loading language model...")
TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
MODEL = TFGPT2LMHeadModel.from_pretrained('gpt2')
TOKENIZER.pad_token = TOKENIZER.eos_token
print("AVA: Systems online. Gradio interface is starting.")
# ---

def generate_response(user_input, chat_history_list):
    """
    Generates a human-like response using the AI model.
    This function is now designed to work with Gradio's ChatInterface.
    """
    
    # Gradio's ChatInterface provides history as a list of lists.
    # We need to format it into a single string for the model.
    chat_history_str = "The following is a conversation with AVA, a friendly and thoughtful AI designed to be indistinguishable from a human.\n\n"
    for human_msg, ai_msg in chat_history_list:
        chat_history_str += f"You: {human_msg}\nAVA: {ai_msg}\n"

    prompt = chat_history_str + f"You: {user_input}\nAVA:"
    
    # Encode the prompt and generate a response
    input_ids = TOKENIZER.encode(prompt, return_tensors='tf')
    
    output_sequences = MODEL.generate(
        input_ids=input_ids,
        max_length=len(input_ids[0]) + 100, # Generate a reasonable length response
        pad_token_id=TOKENIZER.eos_token_id,
        temperature=1.1,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2
    )

    # Decode the response
    response_text = TOKENIZER.decode(output_sequences[0], skip_special_tokens=True)
    
    # Isolate just the new part of the response
    new_response = response_text.replace(prompt.replace("\nAVA:", ""), "").strip()
    
    # Simulate typing
    for char in new_response:
        yield char
        time.sleep(0.02)

# --- Create and Launch the Gradio Web App ---
with gr.Blocks(theme="soft") as app:
    gr.ChatInterface(
        fn=generate_response,
        title="Chat with AVA",
        description="AVA is a conversational AI based on GPT-2. Ask her anything!",
        theme="soft",
        examples=[["Hello, who are you?"], ["What is the meaning of life?"], ["Tell me a short story."]]
    )

# Launch the app. server_name="0.0.0.0" makes it accessible inside Docker.
app.launch(server_name="0.0.0.0", server_port=7860)
