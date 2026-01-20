# Topic 1: Running an LLM

![Race of LLMs](racing.jpg)

### Learning Goals

- Understand how an LLM works at a high level and the role of a tokenizer.

- Have knowledge of the range of LLMs available and the major LLM benchmark datasets on Hugging Face.

- Be able to run tiny/small LLMs on your laptop and small/medium ones on Google Colab on a variety of benchmarks.

- Understand how a chat agent maintains the conversation context, and be able to build a simiple chat agent.

### Tasks

1. Create a Python environment with the following modules installed by conda or pip:

```
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes
```

2. Set up your computing environment with [Hugging Face authorization](COMPLETE_HF_AUTH_GUIDE.html) for Llama 3.2-1B.

3. Verify that your setup is working by running [llama_mmlu_eval.py](llama_mmlu_eval.py) which runs the model on two [MMLU topics](datasets2025.html).

4. Time the code using the time shell command line function. Compare the timings for the following setups:

 1. Using GPU and no quantization.

 2. Using GPU and 4-bit quantization. (Not possible on a MacBook, skip if that is your laptop.)

 3. Using GPU and 8-bit quantization. (Not possible on a MacBook, skip if that is your laptop.)

 4. Using CPU and no quantization.

 5. Using CPU and 4-bit quantization.

5. Modify the code to do the following:

 1. Run on a selection of 10 subjects using [2 other small models](llm_vlm_models_reference.html) in addition to Llama 3.2-1B.

 2. Add timing information to the evaluation summary showing the cycles consumed by each model. Include all of real time, CPU time, and GPU time.

 3. Add an option to the program to make it print out each question, the answer the model gives, and whether the answer is right or wrong.

6. Run the code and create graphs of the results. Can you see any patterns to the mistakes each model makes or do they appear random? Do the all the models make mistakes on the same questions?

7. Repeat the steps above [using Google Colab](COLAB_MMLU_GUIDE.html). In addition to running on the 3 small models from before, try [3 medium-sized models](llm_vlm_models_reference.html). Note that Google Colab has built in integration with Gemini as a coding assistant.

8. Create a chat agent running on your laptop using your favorite model.

 1. Here is a [simple chat agent using Llama 3.2-1B](simple_chat_agent.py) to get you started. You are free to create your chat agent from scratch, but don't use a pre-defined chat agent library - the goal of this exercise is for you to see how a chat agent works from the inside.

 2. The code provided allows the chat history context to grow without limit, which will cause it to eventually fail on long conversations - see the [Llama Chat Context Management Guide](CONTEXT_MANAGEMENT.html) for better approaches. Implement one of them.

 3. Add a flag so that you can turn off the conversation history. Compare how the chat agent performs on a multi-turn conversation when the history is maintained and when it is not.

9. Recommended (optional): Learn how to make your program restartable if it is killed by using the [pickle library](https://docs.python.org/3/library/pickle.html). Test it out by running the program for a while, killing it, and then running it again. If you implement it correctly, then it should pick up where it left off rather than starting from scratch.

10. For the ambitious (optional): MT-Bench is the most popular [multi-turn benchmark](MULTITURN_BENCHMARKS_GUIDE.html). It uses GPT-4 as the automated judge. Get MT-Bench installed and running and test your chat agent. Obtain some chat agents from other students who used different models and compare performance.

### Resources

- [LLM and VLM Reference Guide](llm_vlm_models_reference.html)

- [Datasets](datasets2025.html)

- [Hugging Face Authentication Guide](COMPLETE_HF_AUTH_GUIDE.html)

- [HF Transformers Quickstart](https://huggingface.co/docs/transformers/quicktour)

- [Guide to Running llama_mmlu_eval on Google CoLab](COLAB_MMLU_GUIDE.html)

- [Llama Chat Context Management Guide](CONTEXT_MANAGEMENT.html)

- [Multi-Turn Benchmark Guide](MULTITURN_BENCHMARKS_GUIDE.html)

- [The Illustrated GPT-2 (Visualizing Transformer Language Models), by Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)

- [My conversation with Claude](https://claude.ai/share/2fcb54d9-cd60-43ea-ae69-23fbe889a162)

- [Presentation: Overview of LLMs](01 Running an LLM.pptx)

### Portfolio

Create a subdirectory named "Running an LLM". Include your code and graphs in pdf format comparing accurancy and performance of several different models on several different benchmark datasets, and notes in a markdown file discussing the questions from the tasks.
