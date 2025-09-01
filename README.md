**AI Turing Test Implementation: AVA**

This project contains a conversational AI named AVA (Advanced Virtual Assistant) running inside a Docker container. The goal of this implementation is to create a chatbot that can hold a conversation in a way that is difficult to distinguish from a human, thereby attempting the famous "Turing Test."

**The AI: How it Works**

AVA is powered by OpenAI's **GPT-2**, a large language model. It runs on TensorFlow and leverages the transformers library to handle the complex AI logic.

The core of AVA's "human-like" ability comes from a few key strategies in the turing\_ai.py script:

1.  **Contextual Memory:** AVA remembers the last several turns of the conversation (chat\_history) to provide relevant and coherent responses.

2.  **Creative and Non-Deterministic Responses:** By using a high temperature and nucleus sampling (top\_p), AVA's answers are not always the single "most likely" response. This introduces creativity, randomness, and personality, much like a human.

3.  **Avoids Repetition:** The model is specifically configured to avoid repeating the same phrases, a common giveaway of simpler chatbots.

4.  **Simulated "Thinking":** A slight, randomized delay is added before AVA responds to mimic the time a human might take to type or think.

**How to Run AVA**

**Step 1: Build the Docker Image**

Navigate to the directory containing the Dockerfile, turing\_ai.py, and README.md. Run the build command. This will take some time, as it needs to download the GPT-2 model (around 500MB).

```bash
docker build -t turing-ai .

```

**Step 2: Run the Docker Container**

Once the image is built, you can run it. The -it flags are important as they make the container interactive, allowing you to chat with AVA.

``` bash
docker run -it turing-ai

```

The container will start, initialize the AI, and you can begin your conversation immediately. To stop, simply type exit or quit.
