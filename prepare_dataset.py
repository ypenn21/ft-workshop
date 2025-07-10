import json
import argparse
from datasets import load_dataset

def prepare_sciq_dataset(output_path: str):
    """
    Loads the allenai/sciq dataset from Hugging Face, extracts the 'question'
    and 'correct_answer' fields, and writes them to a JSONL file with
    'input_text' and 'output_text' fields.

    Args:
        output_path (str): The path to save the output JSONL file.
    """
    print("Loading 'allenai/sciq' dataset from Hugging Face...")
    
    # Load all available splits (train, validation, test)
    sciq_dataset = load_dataset("allenai/sciq")
    
    total_examples = 0
    
    print(f"Writing formatted dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for split_name, split_dataset in sciq_dataset.items():
            print(f"Processing split: '{split_name}' ({len(split_dataset)} examples)")
            for example in split_dataset:
                # Create the new format
                formatted_example = {
                    "input_text": example["question"],
                    "output_text": example["correct_answer"]
                }
                # Write the dictionary as a JSON line
                f.write(json.dumps(formatted_example) + "\n")
                total_examples += 1

    print(f"\n✅ Successfully processed and wrote {total_examples} examples.")
    print(f"Dataset saved to '{output_path}'")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Prepare the allenai/sciq dataset for training."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="sciq_dataset.jsonl",
        help="The path to the output JSONL file.",
    )
    args = parser.parse_args()
    
    prepare_sciq_dataset(args.output_file)

if __name__ == "__main__":
    # To run this script, you need to have 'datasets' library installed:
    # pip install datasets
    main()