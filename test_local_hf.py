import argparse
import transformers

TEST_EXAMPLES = [
        "What are good activities for a toddler?",
        "What can we hope to see after rain and sun?",
        "What's the most famous painting by Monet?",
        "Who engineered the Statue of Liberty?",
        'Who were "The Lumières"?',
        ]

# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "Instruction:\n{instruction}\n\nResponse:\n{response}"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

def test_transformers_model(
        model: transformers.GemmaForCausalLM,
        tokenizer: transformers.GemmaTokenizer,
) -> None:
    for prompt in TEST_PROMPTS:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=30)
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{output}\n{'- '*40}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a local Hugging Face Gemma model."
    )
    parser.add_argument(
        "--huggingface_model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the local Hugging Face model and tokenizer.",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.huggingface_model_dir}")
    model = transformers.GemmaForCausalLM.from_pretrained(
        args.huggingface_model_dir,
        local_files_only=True,
        device_map="auto",  # Library "accelerate" to auto-select GPU
    )

    tokenizer = transformers.GemmaTokenizer.from_pretrained(
        args.huggingface_model_dir, local_files_only=True
    )

    print("\n--- Running inference tests ---")
    test_transformers_model(model, tokenizer)
    print("\n--- Tests complete ---")

if __name__ == "__main__":
    main()
