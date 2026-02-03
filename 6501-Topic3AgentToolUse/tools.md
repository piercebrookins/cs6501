Apollo.io
Topic 3: Agent Tool Use
agents using tools

Learning Goals
Know how and why to run an Ollama LLM server

Know how to run large commerical models in LangGraph

Understand how LLMs call tools using the OpenAI format

Understand how to most efficiently define and use LangGraph tools

Tasks
Set up an Ollama server running Llama 3.2-1B  on either your laptop (if it has a GPU) or on CoLab (if not).  Create two copies of the  llama_mmlu_eval.py code from Topic 1 and modify them so that each runs on a single different topic.  Use the command line time shell function  to measure how long each takes to run in real clock time.  Then modify the programs so they use Ollama.  Start the Ollama server running and then time both sequential and parallel execution of the two programs.  For sequential execution, run 

time { python program1.py ; python program2.py }
Next, time them running in parallel:

time { python program1.py & python program2.py & wait; }
Write in the README.md file in your portfolio what you observed.

We will use GPT-4o Mini for the rest of the topic because it is very good at tool use and inexpensive to run (if not completely free) - it should cost less than $1 to complete this topic.  See the full list of OpenAI pricing here.

Sign up, add a payment method, set up your monthly usage limits, and create a secret API key.  Make sure that your secret key never appears in your GitHub portfolio. Evil bots will find it and it will be on the dark web in a  hot second!

Determine how your will pass the secret key to your programs.  Note that because the heavy work is handled in the cloud, you should have no problems running on your laptop even if it has no GPU.  However, you can use Google CoLab if you prefer.  Here is the how to set up your environment for either:

For running on a  laptop: set the key in  ~/.profile by 

export OPENAI_API_KEY="your-actual-key" 
and retrieve it in the code using api_key=os.getenv("OPENAI_API_KEY"). 

For running in Google CoLab, use the  built-in secret manager:

Click the 🔑 key icon in the left sidebar (Secrets)

Click "Add new secret"

Name: OPENAI_API_KEY

Value: your-actual-key`

Toggle notebook access ON

Add to the beginning of your code:

from google.colab import userdata
import os
# Make it available like system env vars
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
Make sure that openai is installed in your Python environment. Run OpenAI GPT4o Mini Test to verify that everything is set up correctly.  If you are running in CoLab, remember to first add the lines above to the beginning of the code.  Be sure that you understand what each of the following two lines are doing - explain them in your README.md portfolio file.		

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: Working!"}],
 max_tokens=5
Download the code sample for manual tool handling and make sure it runs on your setup.  Modify the code to define a calculator tool that includes geometric functions.  Although there are many calculator tools already built into LangGraph, you should define your own from scratch.  To parse the input to your tool use json.loads and to format the output from the tool use json.dumps (if needed). For the calculator itself, you could use ast.literal_eval or numexpr.  Add examples of the system output to your portfolio.  Note: if you find that gpt-4.1-mini insists on trying to do calculations itself instead of using your calculator tool, then explore these strategies for forcing an LLM to use tools and write about it in your README.md.

Download the code sample for LangGraph tool handling and make sure it runs on your set.  Add the calculator you created for task 3 to it.  Create a tool that counts the number of occurences of a letter in a piece of text for answering questions like, "How may s are in Mississippi riverboats?".  Create a third tool of your own invention.  Record the output as you feed different inputs to the system and save it in your portfolio.  For good style, replace the tool execution dispatch code

# Execute the tool
if function_name == "get_weather":
  result = get_weather.invoke(function_args)
else:
  result = f"Error: Unknown function {function_name}"
with something like the following:

# At the top with tool definitions
tools = [get_weather]
tool_map = {tool.name: tool for tool in tools}

# Then in the loop, replace the if/else with:
if function_name in tool_map:
    result = tool_map[function_name].invoke(function_args)
else:
    result = f"Error: Unknown function {function_name}"
Try to create questions that demonstrate multiple tool use for a single question.  For example, "Are there more i's than s's in Mississippi riverboats?" should make the LLM invoke your letter count tool twice in the same turn.  The LLM might or might not use the calculator tool to calculate the difference, but a questions like "What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats" should (if all goes well) invoke the count letter tool twice in the inner loop, takes the responses, and invoke your calculator in the next pass of the outer loop.  Can you create a query that uses all your tools?  Can you get seqential chaining to hit the 5 turn limit in the outer loop?  Add traces of the output to your portfolio and a discussion in your README.md.

Rewrite the program so that runs a single long conversation instead of starting fresh with each call to run_agent.  Instead of a Python loop for the conversation, use LangGraph nodes and edges as you did in the exercises for Agent Orchestration Frameworks.  Include checkpointing and recovery.  Add to your porfolio a Mermaid diagram of the system and examples of a conversation that demonstrates tool use, the conversation context, and recovery.

Question: where is there an opportunity for parallelization in your agent that is not yet being taken advantage of?  Add your answer to the README.md in your portfolio.  (No code for this task.)

Resources
Ollama_guide

OpenAI GPT4o Mini Test

manual-tool-handling.py

langgraph-tool-handling.py

Strategies for forcing an LLM to use tools 

Portfolio
Create a subdirectory in your GitHub portfolio named Topic3Tools and save your programs, each modified version named to indicate its task number and purpose.  Create appropriately named text files saving the outputs from your terminal sessions running the programs.  Create README.md with a table of contents of the directory.

 