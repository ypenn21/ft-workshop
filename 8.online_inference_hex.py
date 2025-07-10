# Online inference

import os
import re
import sys
from google.cloud import aiplatform

def load_config(config_path: str = "config.conf"):
    """
    Parses a shell-style config file and loads variables into the environment.
    This allows Python to read variables from the same config file used by shell scripts.
    """
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Match lines like 'KEY=VALUE', 'KEY="VALUE"', or 'export KEY=VALUE'
                match = re.match(r'^(?:export\s+)?([\w_]+)=(.*)', line)
                if match:
                    key, value = match.groups()
                    # Remove surrounding quotes (single or double)
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

# Load configuration from config.conf into environment variables
load_config()

# --- Configuration from Environment ---
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
ENDPOINT_ID = os.environ.get("HEX_ENDPOINT_ID")  # Use the specific endpoint ID for Hex-LLM

if not all([PROJECT_ID, REGION, ENDPOINT_ID]):
    print("❌ Error: One or more required variables (PROJECT_ID, REGION, HEX_ENDPOINT_ID) not found in config.conf.")
    sys.exit(1)

api_endpoint=f"{REGION}-aiplatform.googleapis.com"

client_options = {"api_endpoint": api_endpoint}
# Initialize client that will be used to create and send requests.
# This client only needs to be created once, and can be reused for multiple requests.
client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
endpoint = client.endpoint_path(
        project=PROJECT_ID, location=REGION, endpoint=ENDPOINT_ID,
        )

TEST_EXAMPLES = [
        "What are good activities for a toddler?",
        "What can we hope to see after rain and sun?",
        "What is the most famous painting by Monet?",
        "Who engineered the Statue of Liberty?",
        'Who were "The Lumières"?',
        ]

# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "Instruction:\nProvide a direct and to the point  answer: {instruction}\n\nResponse:\n{response}"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

def test_vertexai_endpoint(client, endpoint: aiplatform.Endpoint):
    for question, prompt in zip(TEST_EXAMPLES, TEST_PROMPTS):
        instance = {
                "prompt": prompt,
                "max_tokens": 20,
               "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 1,
                "raw_response": True,
                }
        response = client.predict(instances=[instance], endpoint=endpoint)
        output = response.predictions[0].split("Instruction")[0].split("Explanation")[0]
        print(f"{question}\n{output}\n{'- '*80}")


test_vertexai_endpoint(client, endpoint)
