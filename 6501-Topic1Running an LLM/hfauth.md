Apollo.io
Complete Hugging Face Authentication Guide
Table of Contents
Why Authentication is Needed

Installing the Hugging Face CLI

Authentication Methods Overview

Step-by-Step Setup

How Token Detection Works

Environment Variables Explained

Choosing the Right Method

Troubleshooting

Security Best Practices

Quick Reference

Why Authentication is Needed
Some models on Hugging Face (including Llama models from Meta) require:

User authentication - to verify who's downloading the model

License acceptance - to agree to the model's terms of use

Access requests - some models need approval before download

This is Meta's way of tracking usage and ensuring compliance with their license terms.

Installing the Hugging Face CLI
Before you can authenticate, you need to install the huggingface_hub package which provides the CLI tools.

Installation Methods
Option 1: Using pip (Recommended)

pip install huggingface_hub
Option 2: Using pip with --user flag (if you get permission errors)

pip install --user huggingface_hub
Option 3: Using conda/mamba

conda install -c conda-forge huggingface_hub
Option 4: With --break-system-packages (Ubuntu/Debian if needed)

pip install huggingface_hub --break-system-packages
Verify Installation
After installation, verify the installation:

huggingface-cli version
You should see output like:

huggingface-cli, version 0.XX.X
Note: The main command is huggingface-cli, but most subcommands use the shorter hf prefix (like hf auth login, hf whoami).

If Command Not Found
If you get "command not found", make sure pip's bin directory is in your PATH:

Linux/Mac - Temporary:

export PATH="$HOME/.local/bin:$PATH"
Linux/Mac - Permanent:

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
Alternative method if PATH issues persist:

python -m huggingface_hub.commands.huggingface_cli version
Authentication Methods Overview
There are three main ways to authenticate with Hugging Face:

Method	Best For	Token Location	Persistence
hf auth login	Local development, personal use	Saved to disk	Permanent (until logout)
Environment Variable	Servers, CI/CD, Docker	Environment	Session/permanent depending on setup
Explicit token= Argument	Testing, multiple accounts	In code	Per function call
Token Detection Priority
When you use Hugging Face libraries, they check for authentication in this order:

Token passed as token= argument (highest priority - overrides everything)

HF_TOKEN environment variable

HUGGING_FACE_HUB_TOKEN environment variable (legacy)

Saved token from hf auth login

No authentication (will fail for gated models)

Important: If HF_TOKEN is set in your environment, you do not need to pass token= as an argument - the library will automatically use it!

Step-by-Step Setup
Step 1: Create a Hugging Face Account
If you don't have one already:

Go to https://huggingface.co/join

Sign up with email or GitHub account

Verify your email address

Step 2: Create an Access Token
An access token is like a password that lets your computer authenticate with Hugging Face.

Log in to https://huggingface.co

Click your profile picture (top right) → Settings

Go to "Access Tokens" (left sidebar)

Click "New token"

Choose token type:

Read - Download models and datasets (recommended for most users)

Write - Upload models (only if you're sharing models)

Name your token (e.g., "my-laptop", "research-project")

Click "Generate token"

COPY THE TOKEN - You won't be able to see it again!

It looks like: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890

Step 3: Accept Model Licenses
Before you can download gated models like Llama, you must accept their license:

For Llama 3.2-1B:

Go to https://huggingface.co/meta-llama/Llama-3.2-1B

You'll see a message: "Access to this model requires gating"

Click "Request access" or "Accept license"

Fill in the form:

Your intended use case

Agree to Meta's terms

Provide your information

Submit the form

Wait for approval (usually instant, but can take a few minutes)

You'll see a green checkmark when approved: ✓ "You have been granted access to this model"

For Llama 3.2-1B-Instruct:

Same process at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Step 4: Choose Your Authentication Method
Now you need to authenticate your computer. Choose one of the following methods:

Method A: Using hf auth login (Recommended for Local Development)
Best for: Personal laptops, interactive work, Jupyter notebooks

Login
hf auth login
What happens:

Terminal prompts: Token:

Paste your access token (the one you copied from Step 2)

Note: The token won't show as you paste (for security)

Just paste and press Enter

Prompt asks: Add token as git credential? (Y/n)

Type Y and press Enter (recommended)

This saves your token so you don't need to login again

Expected output:

Token is valid (permission: read).
Your token has been saved to /home/username/.cache/huggingface/token
Your token has been saved in your configured git credential helpers (store).
Alternative: Provide token directly in command

hf auth login --token hf_YourTokenHere
Or add the --add-to-git-credential flag to skip the prompt:

hf auth login --token hf_YourTokenHere --add-to-git-credential
Verify Authentication
Check that you're logged in:

hf whoami
Output should show:

username: your_username
email: your_email@example.com
orgs: []
Your Python Code (No Token Needed!)
Once logged in, your Python code needs no token anywhere:

from transformers import AutoModelForCausalLM

# No token argument needed - automatically uses saved token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
✅ This is the cleanest approach for local development!

Logout (if needed)
hf auth logout
Method B: Using Environment Variable (Recommended for Servers/CI/CD)
Best for: Server deployments, Docker containers, CI/CD pipelines, when you need different tokens

Set Environment Variable
Linux/Mac:

export HF_TOKEN="hf_YourTokenHere"
Windows (Command Prompt):

set HF_TOKEN=hf_YourTokenHere
Windows (PowerShell):

$env:HF_TOKEN="hf_YourTokenHere"
Make it permanent (Linux/Mac):

# Add to ~/.bashrc or ~/.zshrc
echo 'export HF_TOKEN="hf_YourTokenHere"' >> ~/.bashrc
source ~/.bashrc
Your Python Code (No Token Argument Needed!)
from transformers import AutoModelForCausalLM

# No token argument needed - automatically uses HF_TOKEN from environment
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
✅ The library automatically detects and uses HF_TOKEN from the environment!

Alternative: Set in Python Script
import os

# Set at the very start of your script (before importing transformers)
os.environ['HF_TOKEN'] = 'hf_YourTokenHere'

from transformers import AutoModelForCausalLM

# No token argument needed - uses HF_TOKEN from os.environ
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
Using .env File (Best Practice)
Create a file named .env in your project directory:

HF_TOKEN=hf_YourTokenHere
IMPORTANT: Add .env to your .gitignore:

.env
*.env
.env.local
In your Python code:

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Now HF_TOKEN is available in environment
from transformers import AutoModelForCausalLM

# Automatically uses HF_TOKEN
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
Install python-dotenv:

pip install python-dotenv
Method C: Explicit Token Argument (For Special Cases)
Best for: Using multiple tokens, testing different accounts, programmatic switching

Pass Token Explicitly
from transformers import AutoModelForCausalLM

# Explicitly pass token - overrides environment variables and saved token
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token="hf_YourTokenHere"
)
Use Case: Multiple Accounts
from transformers import AutoModelForCausalLM

# Use different tokens for different models
model1 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token="hf_PersonalToken"
)

model2 = AutoModelForCausalLM.from_pretrained(
    "my-org/private-model",
    token="hf_OrganizationToken"
)
How Token Detection Works
Automatic Detection - You Asked This!
Question: "If I set HF_TOKEN in the environment, do I also need to provide it as an argument?"

Answer: No! The library automatically detects tokens.

Examples Showing Automatic Detection
Example 1: Environment Variable Only
# In terminal
export HF_TOKEN="hf_YourTokenHere"
# In Python - NO token argument needed!
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B"
    # Automatically uses HF_TOKEN from environment
)
Example 2: After hf auth login
# In terminal
hf auth login
# (paste token once)
# In Python - NO token anywhere!
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B"
    # Automatically uses saved token
)
Example 3: Explicit Override
# Even if HF_TOKEN is set, this overrides it
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token="hf_DifferentToken"  # Uses this instead of environment
)
Testing Token Detection
Use this script to see which token will be used:

import os
from huggingface_hub import HfFolder

print("=== Token Detection Check ===\n")

# Check environment variables
env_token = os.getenv('HF_TOKEN')
legacy_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
print(f"HF_TOKEN env var: {'✓ Set' if env_token else '✗ Not set'}")
print(f"HUGGING_FACE_HUB_TOKEN env var: {'✓ Set' if legacy_token else '✗ Not set'}")

# Check saved token
saved_token = HfFolder.get_token()
print(f"Saved token (from hf auth login): {'✓ Found' if saved_token else '✗ Not found'}")

# Determine which will be used
print("\n=== Which Token Will Be Used ===\n")
if env_token:
    print("✅ HF_TOKEN environment variable")
elif legacy_token:
    print("✅ HUGGING_FACE_HUB_TOKEN environment variable")
elif saved_token:
    print("✅ Saved token from hf auth login")
else:
    print("❌ No token found! Gated models will fail.")
All Functions Use Automatic Detection
Token detection works for all Hugging Face functions:

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import snapshot_download, hf_hub_download

# ALL of these automatically use your token (no token= needed):

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
dataset = load_dataset("meta-llama/some-dataset")
snapshot_download(repo_id="meta-llama/Llama-3.2-1B")
file = hf_hub_download(repo_id="meta-llama/Llama-3.2-1B", filename="config.json")
Environment Variables Explained
Official Variable Names
Hugging Face recognizes multiple environment variable names for backward compatibility:

Variable Name	Status	Priority	Recommendation
HF_TOKEN	✅ Current standard	1st (Highest)	Use this
HUGGING_FACE_HUB_TOKEN	⚠️ Legacy	2nd	Works but outdated
HUGGINGFACE_TOKEN	⚠️ Deprecated	3rd (Lowest)	Avoid
Why Multiple Names?
Hugging Face evolved their naming over time:

Old: HUGGINGFACE_TOKEN

Mid-era: HUGGING_FACE_HUB_TOKEN

Current: HF_TOKEN (shorter, cleaner)

They kept backward compatibility so old code doesn't break.

Priority Example
If you set multiple variables, the first one found wins:

export HF_TOKEN="token_A"
export HUGGING_FACE_HUB_TOKEN="token_B"
# Result: Uses token_A (HF_TOKEN has higher priority)
Other Useful Environment Variables
HF_HOME - Set cache location:

export HF_HOME="/path/to/cache"
HF_HUB_OFFLINE - Use cached models only:

export HF_HUB_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY - Disable analytics:

export HF_HUB_DISABLE_TELEMETRY=1
Choosing the Right Method
Decision Matrix
Scenario	Best Method	Why
Personal laptop	hf auth login	One-time setup, never worry about tokens
Server deployment	Environment variable	Keep tokens in deployment config, not code
Docker container	Environment variable	Pass via -e HF_TOKEN=...
CI/CD pipeline	Environment variable	Use secret management
Multiple accounts	Explicit token=	Programmatically switch tokens
Jupyter notebook	hf auth login	Clean notebooks without tokens
Team development	.env file + .gitignore	Secure, easy to configure
Comparison of Methods
Method 1: hf auth login
Pros:

✅ Cleanest code (no tokens anywhere)

✅ One-time setup

✅ Persistent across sessions

✅ Works for all projects

Cons:

❌ Harder to switch accounts

❌ Not ideal for servers/CI

Example:

hf auth login  # Once
# Clean code forever
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
Method 2: Environment Variable
Pros:

✅ Flexible per-session/project

✅ Great for deployment

✅ Easy to switch tokens

✅ Works with Docker/CI/CD

Cons:

❌ Need to set for each session (unless permanent)

❌ Can forget to set it

Example:

export HF_TOKEN="hf_YourToken"
# Clean code
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
Method 3: Explicit Token
Pros:

✅ Full control

✅ Easy to use different tokens

✅ No environment setup needed

Cons:

❌ Token visible in code

❌ Security risk if committed to git

❌ Repetitive

Example:

# Token in code
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token="hf_YourToken"
)
Troubleshooting
Error: "huggingface-cli: command not found"
Solution 1: Install the package

pip install huggingface_hub
Solution 2: Check PATH

# Find where pip installed packages
pip show huggingface_hub

# Add to PATH (Linux/Mac)
export PATH="$HOME/.local/bin:$PATH"

# Make permanent
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
Solution 3: Use Python module method

python -m huggingface_hub.commands.huggingface_cli auth login
Error: "Access to model requires gating"
You haven't accepted the model's license yet:

Go to the model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B)

Click "Request access"

Fill in the form and wait for approval

Look for green checkmark: ✓ "You have been granted access"

Try downloading again

Error: "Invalid token" or "401 Unauthorized"
Your token is wrong, expired, or has insufficient permissions:

Generate a new token at https://huggingface.co/settings/tokens

Make sure you selected "Read" permissions

Login again: hf auth login

Paste the new token

Error: "Repository not found"
Either:

The model name is misspelled (check exact name on Hugging Face)

You don't have access (need to accept license first)

The model is private/doesn't exist

Token Not Persisting
If hf auth login doesn't save your token:

Manually create token file:

Linux/Mac: ~/.cache/huggingface/token

Windows: C:\Users\YourName\.cache\huggingface\token

Paste your token in the file (just the token, no extra text)

Save and close

Models Downloading to Wrong Location
Check your cache directory:

python -c "from huggingface_hub import HUGGINGFACE_HUB_CACHE; print(HUGGINGFACE_HUB_CACHE)"
Change cache location:

export HF_HOME="/path/to/your/cache"
Security Best Practices
✅ DO:
Use .env files with .gitignore

# .env file
HF_TOKEN=hf_YourTokenHere

# .gitignore
.env
*.env
Set in shell profile for permanent use

# ~/.bashrc or ~/.zshrc
export HF_TOKEN="hf_YourTokenHere"
Use hf auth login for personal machines

hf auth login  # Tokens saved securely
Use environment variables in deployment

# Docker
docker run -e HF_TOKEN="hf_Token" myimage

# Kubernetes
# Store in secrets, not ConfigMaps
❌ DON'T:
Never hardcode tokens in source code

# ❌ BAD - Don't do this!
token = "hf_AbCdEfGhIjKlMnOpQrStUvWxYz"
Never commit tokens to git

# ❌ BAD - Don't do this!
git add .env
git commit -m "Added token"
Never share tokens

Tokens are like passwords

Each person should have their own

Never use Write tokens unless needed

Use Read tokens for downloading

Only use Write for uploading

Token Permissions
When creating tokens, use minimum necessary permissions:

Read - For downloading models (most common)

Write - Only for uploading models

Fine-grained - For specific repositories only

Regenerating Compromised Tokens
If your token is exposed:

Go to https://huggingface.co/settings/tokens

Delete the compromised token

Generate a new token

Update your configuration

Managing Downloaded Models
Where Models Are Cached
Default locations:

Linux/Mac: ~/.cache/huggingface/hub/

Windows: C:\Users\YourName\.cache\huggingface\hub\

View Cache
hf cache info
Shows all downloaded models and their sizes.

Delete Cached Models
Delete specific model:

hf cache delete --repo-id meta-llama/Llama-3.2-1B
Or manually:

rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B
Change Cache Location
export HF_HOME="/path/to/new/cache"
Make it permanent by adding to ~/.bashrc or ~/.zshrc.

Complete Example Workflow
Here's the complete process from installation to running code:

# 1. Install Hugging Face Hub
pip install huggingface_hub transformers torch

# 2. Verify installation
huggingface-cli version

# 3. Login to Hugging Face (recommended for local dev)
hf auth login
# (Paste your token when prompted)

# 4. Verify login
hf whoami

# 5. Accept Llama license
# Visit: https://huggingface.co/meta-llama/Llama-3.2-1B
# Click "Request access" and fill out form

# 6. Run your Python code
python your_script.py
First run: Downloads ~2.5GB model to cache
Subsequent runs: Uses cached model, no download needed

Quick Reference
Essential Commands
# Installation
pip install huggingface_hub

# Check version
huggingface-cli version

# Login
hf auth login

# Check who you're logged in as
hf whoami

# Check authentication status
hf auth status

# Logout
hf auth logout

# View cache
hf cache info

# Delete from cache
hf cache delete --repo-id MODEL_NAME

# Get help
hf --help
hf auth --help
Environment Variables
# Set token (recommended)
export HF_TOKEN="hf_YourTokenHere"

# Legacy (still works)
export HUGGING_FACE_HUB_TOKEN="hf_YourTokenHere"

# Set cache location
export HF_HOME="/path/to/cache"

# Offline mode
export HF_HUB_OFFLINE=1
Python Code Patterns
# Pattern 1: No token needed (after hf auth login or HF_TOKEN set)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Pattern 2: Explicit token (overrides environment)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token="hf_YourTokenHere"
)

# Pattern 3: Set in code (before importing transformers)
import os
os.environ['HF_TOKEN'] = 'hf_YourTokenHere'
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Pattern 4: Using .env file
from dotenv import load_dotenv
load_dotenv()  # Loads HF_TOKEN from .env
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
Common Model Access URLs
Llama 3.2-1B: https://huggingface.co/meta-llama/Llama-3.2-1B

Llama 3.2-1B-Instruct: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Llama 3.2-3B: https://huggingface.co/meta-llama/Llama-3.2-3B

OLMo 3-7B: https://huggingface.co/allenai/OLMo-3-1125-7B

Summary Checklist

Install: pip install huggingface_hub


Verify: huggingface-cli version


Create Hugging Face account at https://huggingface.co/join


Generate token at https://huggingface.co/settings/tokens (Read permission)


Accept model license at model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B)


Choose authentication method:

Local dev: Run hf auth login

Server/Docker: Set export HF_TOKEN="..."

Flexible: Use .env file with python-dotenv


Verify: hf whoami


Run your code - models download automatically on first use!

Key Takeaways
You do NOT need token= argument if using hf auth login or HF_TOKEN environment variable

HF_TOKEN is the current standard (not HUGGING_FACE_HUB_TOKEN)

Token detection is automatic - library checks environment and saved tokens

For local dev: Use hf auth login (cleanest approach)

For servers: Use HF_TOKEN environment variable

For security: Use .env files and add to .gitignore

Must accept license on model page before downloading

First download caches - subsequent uses are instant

That's everything you need to know about Hugging Face authentication! 🚀