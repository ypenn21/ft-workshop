# Online inference

import os
import sys
import argparse
from google.cloud import aiplatform
from utils import load_config

def test_vertexai_endpoint(client, endpoint_path: str, test_examples: list, test_prompts: list):
    """Sends prediction requests to a Vertex AI endpoint and prints the results."""
    for question, prompt in zip(test_examples, test_prompts):
        instance = {
                "prompt": prompt,
                "max_tokens": 20,
               "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 1,
                "raw_response": True,
                }
        response = client.predict(instances=[instance], endpoint=endpoint_path)
        output = response.predictions[0].split("Instruction")[0].split("Explanation")[0]
        print(f"{question}\n{output}\n{'- '*80}")

def main():
    """Main function to run online inference."""
    parser = argparse.ArgumentParser(
        description="Run online inference for a deployed model on a Vertex AI Hex-LLM endpoint."
    )
    parser.add_argument(
        "endpoint_id",
        type=str,
        help="The ID of the Vertex AI Endpoint to use for inference.",
    )
    args = parser.parse_args()

    # Load configuration from config.conf into environment variables
    load_config()

    # --- Configuration from Environment ---
    PROJECT_ID = os.environ.get("PROJECT_ID")
    REGION = os.environ.get("REGION")
    ENDPOINT_ID = args.endpoint_id

    if not all([PROJECT_ID, REGION, ENDPOINT_ID]):
        print("❌ Error: PROJECT_ID and REGION must be in config.conf, and endpoint_id must be provided as a command-line argument.")
        sys.exit(1)

    api_endpoint=f"{REGION}-aiplatform.googleapis.com"

    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    endpoint_path = client.endpoint_path(
            project=PROJECT_ID, location=REGION, endpoint=ENDPOINT_ID,
            )

    test_examples = [
            "What are good activities for a toddler?",
            "What can we hope to see after rain and sun?",
            "What is the most famous painting by Monet?",
            "Who engineered the Statue of Liberty?",
            'Who were "The Lumières"?',
            ]

    # Prompt template for the training data and the finetuning tests
    prompt_template = "Instruction:\nProvide a direct and to the point  answer: {instruction}\n\nResponse:\n{response}"

    test_prompts = [
            prompt_template.format(instruction=example, response="")
            for example in test_examples
            ]

    test_vertexai_endpoint(client, endpoint_path, test_examples, test_prompts)

if __name__ == "__main__":
    main()
