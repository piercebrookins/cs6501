Apollo.io
Topic 2: Agent Orchestration Frameworks
agents talking

Learning Goals
Understand LangGraph concepts, including  nodes, edges, conditional edges, routers, parallelism, and checkpoints

Be able to build a LangGraph multi-agent chat system

Tasks
Read the LangChain Graph API Overview.  

Download and run langgraph_simple_llama_agent.py using requirements.txt on either your laptop (if it has a GPU) or CoLab.  Read the code and think about how it wraps a Hugging Face LLM and how defines nodes, routers, and conditional edges.  Modify the code so that if the input is the word "verbose" then each node prints tracing information to stdout, and if the input is "quiet" the tracing information is not printed.

See what happens when you give the program an empty input.  Record what it does.  Try another empty input.  What happens this time?  What does this reveal about less large and sophisticated LLMs such as the one here, llama-3.2b-instruct?  Modify the code so that an empty input is never passed to the LLM. Instead of adding a loop that ignores empty input, get into the spirit of LangChain and modify the input_node get_user_input node and router function so there is a 3-way conditional branch out of get_user_input node, with one edge going back to itself. 

Modify the code again so that the output edge from get_user_input that continues on the LLM instead goes to a node that simply passes the input onto both a node for Llama and a node for your choice of Qwen model. The models should run in parallel.  The node that accepts the inputs from both models should print out both results.

Modify the code so that instead of running both models in parallel, only one of them is run.  If the user's input begins with the words "Hey Qwen", then it should go to Qwen, otherwise to Llama.

Note that the program does not maintain a chat history context.  Modify it so that it does using the Message API.  The roles supported by the API are system, human (or user), ai (or assistant), and tool (or function).  See the Graph API Overview.  Disable the ability to use Qwen and test your code to make sure that it is working.

Now you are going to integrate the chat history with the ability to switch between Llama and Qwen. The challenge here is that there are three entities involved (you the human, Llama, and Qwen) but a chat history only has the roles user, assistant, system, and tool.  This can be handled by using the "user" role for both the human and the other LLM by adding their names to what each says.  For example, consider the dialog:

(user) What is the best ice cream flavor?

(Llama) There is no one best flavor, but the most popular is vanilla.

(user) Hey Qwen, what do you think?
At this point, Qwen should be passed a history that looks like:

[ {role: "user", content: "Human: What is the best ice cream flavor?"},
  {role: "user", content: "Llama: There is no one best flavor, but the most popular is vanilla."} ]
Suppose the conversation continues:

(Qwen) No way, chocolate is the best!
(user) I agree.
At this point, Llama should be passed a history that looks like:

[ {role: "user", content: "Human: What is the best ice cream flavor?"},
  {role: "assistant", content: "Llama: There is no one best flavor, but the most popular is vanilla."},
  {role: "user", content: "Qwen: No way, chocolate is the best!"},
  {role: "user", content: "Human: I agree."} ]
You will also need to add a system prompt for each LLM, stating who the participants are, modfied according to whether the prompt is for Llama or Qwen.  Record some interesting conversations. 

One of the most important features of LangGraph is that it supports checkpointing and crash recovery. (You can also use checkpointing and recovery manually in order to send your agent back in time, but we will focus here on its use to provide recovery when long-running agent processes crash.)  Read this document on LangGraph Crash Recovery and discuss it with your neighbors in class.  Then modify the multi-agent chat program you just wrote so that you can kill it in the middle of conversation and restart it with nothing lost.

Resources
LangChain Graph API Overview

Henry's gpt_with_tools

langgraph_simple_llama_agent.py and requirements.txt

LangGraph Crash Recovery

Portfolio
Create a subdirectory in your GitHub portfolio named Topic2Frameworks and save your programs, each modified version named to indicate its task number and purpose.  Create appropriately named text files saving the outputs from your terminal sessions running the programs.  Create README.md with a table of contents of the directory.

Note: to make it easier for me to review your work, please create a single repository for the entire course, with subdirectores for each topic.  Email me and the TA a link to  your repository.