import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import time
import random

def initialize_ai():
    """Loads the pre-trained GPT-2 model and tokenizer from the cache."""
    print("AVA: Booting up...")
    print("AVA: Loading language model from cache. This might take a moment...")
    
    # Load the tokenizer and model. These were downloaded when the Docker image was built.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    # Set a padding token to handle inputs of different lengths.
    tokenizer.pad_token = tokenizer.eos_token
    print("AVA: Systems online. I'm ready to talk.")
    print("-" * 50)
    return tokenizer, model

def generate_response(tokenizer, model, user_input, chat_history):
    """Generates a human-like response using the AI model."""
    
    # Combine the user's new message with the past conversation for context
    prompt = chat_history + "You: " + user_input + "\nAVA:"
    
    # Encode the prompt into a format the model understands (tensor)
    input_ids = tokenizer.encode(prompt, return_tensors='tf')

    # Generate a response using the model
    # These parameters are tuned to make the conversation feel more natural and less robotic.
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=150,  # Max length of the response
        pad_token_id=tokenizer.eos_token_id,
        temperature=1.1,      # Makes the output more creative and less predictable
        top_k=50,             # Considers the top 50 most likely next words
        top_p=0.95,           # Nucleus sampling for more diverse responses
        no_repeat_ngram_size=2 # Prevents the model from repeating phrases
    )

    # Decode the generated sequence of numbers back into text
    response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # Isolate just the new response from the model
    new_response = response_text.replace(prompt.replace("\nAVA:", ""), "").strip()
    
    # Simulate a "typing" delay to feel more human
    time.sleep(random.uniform(0.5, 1.5))

    return new_response

def chat():
    """Main function to run the conversational loop."""
    tokenizer, model = initialize_ai()
    
    # Start with a predefined persona or context
    chat_history = "The following is a conversation with AVA, a friendly and thoughtful AI. AVA is designed to be indistinguishable from a human in a blind test. It is curious, sometimes humorous, and has a wide range of knowledge.\n\n"
    
    print("Welcome! You are now talking to AVA. Type 'exit' or 'quit' to end the conversation.")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("AVA: It was nice talking to you. Goodbye!")
            break

        # Generate and print the AI's response
        ava_response = generate_response(tokenizer, model, user_input, chat_history)
        print(f"AVA: {ava_response}")

        # Update the chat history to maintain context for the next turn
        chat_history += f"You: {user_input}\nAVA: {ava_response}\n"
        
        # To prevent the context from getting too long, we can trim it if needed
        if len(chat_history) > 2048:
            chat_history = chat_history[-2048:]


if __name__ == "__main__":
    chat()
