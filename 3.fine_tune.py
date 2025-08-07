"""
Set up Keras JAX backend
Import JAX and run a sanity check on TPU. TPUv6e-4 host offers 4 TPU cores with 32GB of memory each.


Here’s a summary of the training (fine-tuning) process as implemented in 3.fine_tune.py:

Setup and Configuration

Loads configuration and environment variables (such as dataset name, model name, batch size, etc.) from a config file.
Sets up the Keras backend to use JAX, optimized for TPU training.
Distributed Model Loading

Sets up a device mesh and sharding strategy to distribute the model’s weights across multiple TPUs for efficient parallel training.
Loads the base language model (e.g., a Gemma model) with specified sharding.
Pre-finetuning Evaluation

Runs inference with the base model on test prompts to show how the model performs before fine-tuning.
Preparing Training Data

Loads the dataset (in JSONL format), converts each data entry into a prompt/response format, shuffles the data, and selects the desired percentage for training.
Fine-tuning Setup

Optionally enables LoRA (Low-Rank Adaptation) for efficient parameter updating (though this may not be compatible with sharding in this implementation).
Sets the input sequence length and configures the optimizer (AdamW with weight decay, excluding biases and normalization layers from decay).
Model Training

Compiles the model with loss and metrics (including perplexity).
Trains the model on the prepared dataset for a specified number of epochs and batch size.
Post-finetuning Evaluation

Runs inference again on the same test prompts to show improvements after fine-tuning.
Saving the Model

Saves the fine-tuned weights, vocabulary, and Keras preset to disk for future use or deployment.
This process leverages TPUs for scale, supports advanced distribution and sharding, and includes before/after evaluation to demonstrate the effect of fine-tuning. Let me know if you’d like details on any specific step!

what is the best efficient EPOCHS to set this to? what does equal to in terms of hours or minutes?
The optimal number of EPOCHS for fine-tuning depends on several factors, including dataset size, model size, and the desired trade-off between performance and training time. In many practical cases:

For small to medium datasets, 3–5 epochs is common.
For larger datasets or when overfitting is not a concern, 10–20 epochs can be used.
If you see the model start to overfit (validation loss increases while training loss decreases), you should reduce the number of epochs or use early stopping.
For instructional or demonstration purposes, 1–3 epochs is often sufficient.
In this specific repo (ft-workshop):

The default value is not hardcoded in the script; it is expected from config.conf.
The script is designed for distributed training on TPUs, which is very efficient.
How long does an epoch take?
The time per epoch depends on:

Dataset size: More data = longer epoch.
Batch size: Larger batch = fewer steps per epoch, but more memory usage.
Model size: Larger models take longer per batch.
Hardware: More/faster TPUs = faster training.
Example estimation (for TPUs, as in your script):
Small dataset (a few thousand examples), batch size 32–128, 8 TPU cores:
One epoch can take from under a minute to several minutes.
Larger dataset (hundreds of thousands of examples):
One epoch could take 10–30 minutes.
Rule of thumb for quick experimentation:

Set EPOCHS=3 for initial runs.
Monitor training/validation loss:
If the model is still improving, increase epochs.
If it plateaus or overfits, stop earlier.
To get a precise time:
Run with EPOCHS=1 and measure the time taken for that epoch.
Multiply by the desired number of epochs.
Summary:

Start with 3–5 epochs for efficiency.
One epoch may take from a minute to 30 minutes, depending on dataset and hardware.
Measure with a single epoch run for an exact estimate.
Would you like help modifying your config or want a script to time a single epoch?

what does epoch mean exactly?
In machine learning, an epoch is one complete pass through the entire training dataset by the learning algorithm.

During one epoch, the model sees every example in the dataset once and updates its weights accordingly (often in smaller batches).
Training usually involves multiple epochs, so the model processes the dataset multiple times, each time learning and adjusting to minimize error.
For example, if you have 1,000 training samples and use a batch size of 100, it will take 10 batches to complete 1 epoch. If you train for 5 epochs, the model will see each sample 5 times (in different orders, if shuffled).
Summary:
An epoch = one full cycle through all the training data. Multiple epochs help the model learn better by allowing it to adjust its weights based on the data several times.
"""

import jax

print("TPU devices:\n",jax.devices(),"\n")

NUM_TPUS=jax.device_count()

import random
import os
import json
import sys
from utils import load_config

# Load configuration from config.conf into environment variables
load_config()

# Check for required credentials and configuration
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
DATASET_NAME = os.environ.get("DATASET_NAME")

EPOCHS = os.environ.get("EPOCHS")
BATCH_PER_TPU = os.environ.get("BATCH_PER_TPU")

if not all([KAGGLE_USERNAME, KAGGLE_KEY, MODEL_NAME, DATASET_NAME, EPOCHS, BATCH_PER_TPU]):
    print("❌ Error: KAGGLE_USERNAME, KAGGLE_KEY, MODEL_NAME, EPOCHS, BATCH_PER_TPU and DATASET_NAME must be set in config.conf.")
    sys.exit(1)

# The Keras 3 distribution API is only implemented for the JAX backend for now.
os.environ["KERAS_BACKEND"] = "jax"
# Pre-allocate 90% of TPU memory to minimize memory fragmentation and allocation
# overhead
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

"""
A few configuration parameters
"""
# MODEL_NAME is now read from config.conf
# Deduce model size from name format: "gemma2[_instruct]_{2b,9b}_en"
#MODEL_SIZE = MODEL_NAME.split("_")[-2]
#assert MODEL_SIZE in ("2b", "7b")

# Dataset
DATASET_PATH = f"{DATASET_NAME}.jsonl"

# Finetuned model
FINETUNED_MODEL_DIR = "/mnt/content/finetuned"
FINETUNED_KERAS_DIR = "/mnt/content/finetuned_keras"
FINETUNED_WEIGHTS_PATH = f"{FINETUNED_MODEL_DIR}/model.weights.h5"
FINETUNED_VOCAB_PATH = f"{FINETUNED_MODEL_DIR}/vocabulary.spm"

#EPOCHS=10
#BATCH_PER_TPU=4
BATCH_SIZE=BATCH_PER_TPU*NUM_TPUS

"""
Load model

To load the model with the weights and tensors distributed across TPUs, first create a new DeviceMesh.
DeviceMesh represents a collection of hardware devices configured for distributed computation
and was introduced in Keras 3 as part of the unified distribution API.
"""

import keras
keras.utils.set_random_seed(42)
# Run inferences at half precision
#keras.config.set_floatx("bfloat16")
# Train at mixed precision (enable for large batch sizes)
keras.mixed_precision.set_global_policy("mixed_bfloat16")
import keras_hub

# Create a device mesh with (1, 8) shape so that the weights are sharded across
# all 4 TPUs.
device_mesh = keras.distribution.DeviceMesh(
            (1, NUM_TPUS),
            ["batch", "model"],
            devices=keras.distribution.list_devices())


"""
LayoutMap from the distribution API specifies how the weights and tensors should be sharded or replicated,
using the string keys, for example, token_embedding/embeddings below,
which are treated like regex to match tensor paths. Matched tensors are sharded with model dimensions (4 TPUs);
others will be fully replicated.
"""

model_dim = "model"

layout_map = keras.distribution.LayoutMap(device_mesh)

# Weights that match 'token_embedding/embeddings' will be sharded on 4  TPUs
layout_map["token_embedding/embeddings"] = (model_dim, None)
# Regex to match against the query, key and value matrices in the decoder
# attention layers
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (model_dim, None, None)

layout_map["decoder_block.*attention_output.*kernel"] = (model_dim, None, None)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)

"""
ModelParallel allows you to shard model weights or activation tensors across all devices on the DeviceMesh.
In this case, some of the Gemma 7B model weights are sharded across 4 TPU chips
according to the layout_map defined above. Now load the model in the distributed way.
"""

model_parallel = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch")

keras.distribution.set_distribution(model_parallel)
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(f"{MODEL_NAME}")
gemma_lm.summary()

"""
Now verify that the model has been partitioned correctly. Let's take decoder_block_1 as an example.
"""

decoder_block_1 = gemma_lm.backbone.get_layer('decoder_block_1')
print(type(decoder_block_1))
for variable in decoder_block_1.weights:
      print(f'{variable.path:<58}  {str(variable.shape):<16}  {str(variable.value.sharding.spec)}')

"""
Inference before finetuning
"""

TEST_EXAMPLES = [
        #"Lizzy has to ship 540 pounds of fish that are packed into 30-pound crates. If the shipping cost of each crate is $1.5, how much will Lizzy pay for the shipment?",
        #"A school choir needs robes for each of its 30 singers. Currently, the school has only 12 robes so they decided to buy the rest. If each robe costs $2, how much will the school spend?",
        "Peter has 25 apples to sell. He sells the first 10 for $1 each, the second lot of 10 for $0.75 each and the last 5 for $0.50 each. How much money does he make?",
        "Bea has $40. She wants to rent a bike for $4/hour. How many hours can she ride the bike?",
        ]

# Prompt template for the training data and the finetuning tests
PROMPT_TEMPLATE = "user: {instruction}\nmodel: {response}\n"

TEST_PROMPTS = [
        PROMPT_TEMPLATE.format(instruction=example, response="")
        for example in TEST_EXAMPLES
        ]

gemma_lm.compile(sampler="greedy")

print ("Before fine-tuning:\n")

for test_example in TEST_EXAMPLES:
        response = gemma_lm.generate(test_example, max_length=256)
        output = response[len(test_example) :]
        print(f"{test_example}\n{output!r}\n")

"""
Download and prepare dataset
"""

print("\nDowloading and preparing fine-tuning dataset...\n")

#os.system(f"wget -nv -nc -O {DATASET_PATH} {DATASET_URL}")

def generate_training_data(training_ratio: int = 100) -> list[str]:
        assert 0 < training_ratio <= 100
        data = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                features = json.loads(line)
                # Format the data into the prompt template
                data.append(PROMPT_TEMPLATE.format(
                    instruction=features["input_text"],
                    response=features["output_text"]
                ))
        print("Shuffling training data...")
        random.shuffle(data)

        total_data_count = len(data)
        training_data_count = total_data_count * training_ratio // 100
        print(f"Training examples: {training_data_count}/{total_data_count}")
        return data[:training_data_count]

# Limit to 10% for test purposes

training_data = generate_training_data(training_ratio=100)


"""
Fine-tune with or without LoRA

LoRA is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks
by freezing the full weights of the model and inserting a smaller number of new trainable weights into the model.
Basically LoRA reparameterizes the larger full weight matrices by 2 smaller low-rank matrices AxB
to train and this technique makes training much faster and more memory-efficient.
"""

print ("\nFine-tuning...\n")

# Enable LoRA for the model and set the LoRA rank to 4.
#gemma_lm.backbone.enable_lora(rank=4) #DOES NOT SEEM TO WORK WITH SHARDING
#gemma_lm.summary()

# Limit the input sequence length to 128 to control memory usage.
gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.001,)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# Compile and train
gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras_hub.metrics.Perplexity(from_logits=True)],
        sampler="greedy")

gemma_lm.fit(training_data, epochs=EPOCHS, batch_size=BATCH_SIZE)

"""
Inference after fine-tuning
"""

print ("After fine-tuning:\n")
for prompt in TEST_PROMPTS:
        output = gemma_lm.generate(prompt, max_length=256)
        print(f"{output}\n{'- '*40}")

# Finetuned model

print ("\nSaving fine-tuned model weights...\n")

# Make sure the directory exists
os.system("mkdir -p "+FINETUNED_MODEL_DIR)
os.system("mkdir -p "+FINETUNED_KERAS_DIR)

gemma_lm.save_weights(FINETUNED_WEIGHTS_PATH, overwrite=True)

gemma_lm.preprocessor.tokenizer.save_assets(FINETUNED_MODEL_DIR)

print ("\nModel weights saved.\n")

print ("\nSaving fine-tuned model preset in keras format...\n")
gemma_lm.save_to_preset(FINETUNED_KERAS_DIR)
print ("\nDone.\n")
