Apollo.io
Topic 1: Running an LLM 
Race of LLMs

Learning Goals
Understand how an LLM works at a high level and the role of a tokenizer.

Have knowledge of the range of LLMs available and the major LLM benchmark datasets on Hugging Face.

Be able to run tiny/small LLMs on your laptop and small/medium ones on Google Colab on a variety of benchmarks.

Understand how a chat agent maintains the conversation context, and be able to build a simiple chat agent.

Tasks
Create a Python environment with the following modules installed by conda or pip:

pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes
Set up your computing environment with Hugging Face authorization for Llama 3.2-1B.

Verify that your setup is working by running llama_mmlu_eval.py which runs the model on two MMLU topics.  

Time the code using the time shell command line function. Compare the timings for the following setups:

Using GPU and no quantization.

Using GPU and 4-bit quantization. (Not possible on a MacBook, skip if that is your laptop.)

Using GPU and 8-bit quantization. (Not possible on a MacBook, skip if that is your laptop.)

Using CPU and no quantization.

Using CPU and 4-bit quantization.

Modify the code to do the following:

Run on a selection of 10 subjects using 2 other small models in addtion to Llama 3.2-1B.  

Add timing information to the evaluation summary showing the cycles consumed by each model.  Include all of real time, CPU time, and GPU time.

Add an option to the program to make it print out each question, the answer the model gives, and whether the answer is right or wrong. 

Run the code and create graphs of the results.  Can you see any patterns to the mistakes each model makes or do they appear random?  Do the all the models make mistakes on the same questions?  

Repeat the steps above using Google Colab.  In addition to running on the 3 small models from before, try 3 medium-sized models.  Note that Google Colab has built in integration with Gemini as a coding assistant.

Create a chat agent running on your laptop using your favorite model.  

Here is a simple chat agent using Llama 3.2-1B to get you started.  You are free to create your chat agent from scratch, but don't use a pre-defined chat agent library - the goal of this exercise is for you to see how a chat agent works from the inside.

The code provided allows the chat history context to grow without limit, which will cause it to eventually fail on long conversations - see the Llama Chat Context Management Guide for better approaches.  Implement one of them.

Add a flag so that you can turn off the conversation history. Compare how the chat agent performs on a multi-turn conversation when the history is maintained and when it is not.

Recommended (optional): Learn how to make your program restartable if it is killed by using the pickle library.  Test it out by running the program for a while, killing it, and then running it again.  If you implement it correctly, then it should pick up where it left off rather than starting from scratch.

For the ambitious (optional): MT-Bench is the most popular multi-turn benchmark.  It uses GPT-4 as the automated judge.  Get MT-Bench installed and running and test your chat agent.  Obtain some chat agents from other students who used different models and compare performance.

Resources
LLM and VLM Reference Guide

Datasets

Hugging Face Authentication Guide

HF Transformers Quickstart 

Guide to Running llama_mmlu_eval on Google CoLab 

Llama Chat Context Management Guide

Multi-Turn Benchmark Guide

The Illustrated GPT-2 (Visualizing Transformer Language Models), by Jay Alammar 

My conversation with Claude

Presentation: Overview of LLMs

Portfolio
Create a subdirectory named "Running an LLM". Include your code and graphs in pdf format comparing accurancy and performance of several different models on several different benchmark datasets, and notes in a markdown file discussing the questions from the tasks.