import pandas as pd
import random
import os
import json
import argparse
from typing import List, Dict, Union, Optional
import warnings

# For local LLM
try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
except ImportError:
    Llama = None
    hf_hub_download = None
    warnings.warn(
        "llama-cpp-python or huggingface_hub not found. "
        "LLM-based augmentation will not be available unless these are installed."
    )


# --- Configuration for Data Paths ---
TRAIN_ORIGINAL_CSV = "train_original.csv"
VAL_CSV = "val.csv"
TRAIN_AUGMENTED_CSV = "train.csv" # The final training CSV for open_clip

# --- Manual Synonym Map (Option 1) ---
MANUAL_SYNONYM_MAP_FILE = "manual_synonym_map.json"
DEFAULT_MANUAL_SYNONYM_MAP = {
    "cat": ["feline", "kitty", "house cat"],
    "dog": ["canine", "puppy", "hound"],
    "car": ["automobile", "vehicle", "auto"],
    "house": ["home", "residence", "dwelling"],
    "man": ["person", "male", "gentleman"],
    "woman": ["person", "female", "lady"],
    "running": ["jogging", "sprinting", "dashing"],
    "big": ["large", "huge", "enormous"],
    "small": ["tiny", "little", "petite"],
}

# --- Local LLM Configuration (Option 2) ---
LLM_MODEL_REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf" 
LLM_MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf" 
LLM_MODEL_LOCAL_PATH = "local_llm_models"

class LLMParaphraser:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMParaphraser, cls).__new__(cls)
            cls._instance.llm = None
            cls._instance.model_path = None
        return cls._instance

    def __init__(self, model_repo_id: str, model_filename: str, local_path: str):
        if self.llm is not None and self.model_path == os.path.join(local_path, model_filename):
            return # Already initialized

        if Llama is None or hf_hub_download is None:
            raise ImportError(
                "llama-cpp-python and huggingface_hub are required for LLM-based augmentation. "
                "Please install them: pip install llama-cpp-python huggingface_hub"
            )

        model_full_path = os.path.join(local_path, model_filename)
        os.makedirs(local_path, exist_ok=True)

        if not os.path.exists(model_full_path):
            print(f"Downloading LLM model '{model_filename}' from Hugging Face Hub...")
            hf_hub_download(
                repo_id=model_repo_id,
                filename=model_filename,
                local_dir=local_path,
                local_dir_use_symlinks=False, # Important for local persistent storage
            )
            print("Download complete.")
        else:
            print(f"LLM model '{model_filename}' already exists at {model_full_path}.")

        self.model_path = model_full_path
        # Initialize Llama model
        # n_gpu_layers=-1 will attempt to offload all layers to GPU if CUDA/Metal is available
        # n_ctx: context window size. 2048 or 4096 usually sufficient for paraphrasing.
        # verbose=False: suppress llama.cpp loading logs
        self.llm = Llama(model_path=self.model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
        print(f"Local LLM '{model_filename}' loaded successfully.")

    def paraphrase(self, caption: str, num_variations: int = 1) -> List[str]:
        """
        Generates paraphrases using the loaded local LLM.
        """
        # The prompt is crucial for good results
        prompt_template = """
        You are an AI assistant specialized in text variations/synonyms
        Concisely paraphrase (similar amount of words) the following description of an image
        Provide {num_variations} distinct variations
        Each variation should be on a new line.

        Sentence: "{caption}"

        Variations:
        """
        
        prompt = prompt_template.format(caption=caption.strip(), num_variations=num_variations)

        paraphrases = []
        try:
            # Llama.create_completion is for older models. Llama.create_chat_completion is for instruction-tuned.
            # Phi-3-mini-4K-Instruct is instruction-tuned, so we should use chat completion.
            # Using messages API for better results with instruction-tuned models
            messages = [
                {"role": "system", "content": "You are a helpful assistant that paraphrases sentences"},
                {"role": "user", "content": prompt}
            ]
            
            # max_tokens can be adjusted based on expected output length
            # temperature controls creativity (0.0 for deterministic, higher for more varied)
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=256 # Enough for a few short paraphrases
            )
            
            raw_output = response['choices'][0]['message']['content'].strip()
            
            # Split the raw output into individual paraphrases
            # LLMs might add bullet points, numbers etc., so we try to clean
            lines = raw_output.split('\n')
            for line in lines:
                cleaned_line = line.strip()
                # Remove common list prefixes if LLM added them
                if cleaned_line and (
                    cleaned_line.startswith('- ') or
                    cleaned_line.startswith('* ') or
                    cleaned_line.startswith('1. ') or
                    cleaned_line.startswith('2. ') or
                    cleaned_line.startswith('A. ')
                ):
                    cleaned_line = cleaned_line[cleaned_line.find(' ') + 1:].strip()
                if cleaned_line: # Ensure it's not an empty string after cleaning
                    paraphrases.append(cleaned_line)
                    if len(paraphrases) >= num_variations: # Stop if we have enough
                        break
            
            # If the LLM didn't produce enough, fill with original or simple variations
            while len(paraphrases) < num_variations:
                paraphrases.append(f"A different way of saying: {caption}") # Fallback
                
        except Exception as e:
            print(f"Error during LLM paraphrase: {e}")
            # Fallback to a simple variation if LLM fails
            paraphrases = [f"A variation of: {caption} (Error fallback {i+1})" for i in range(num_variations)]
        
        print(f"Generated {len(paraphrases)} paraphrases for: '{caption}'")
        return paraphrases[:num_variations] # Ensure we return exactly num_variations

# Helper functions (from previous version, no change needed)
def load_synonym_map(filepath: str) -> Dict[str, List[str]]:
    """Loads a synonym map from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_synonym_map(synonym_map: Dict[str, List[str]], filepath: str):
    """Saves a synonym map to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(synonym_map, f, indent=4)
    print(f"Saved synonym map to {filepath}")

def apply_manual_synonym_augmentation(
    caption: str,
    synonym_map: Dict[str, List[str]],
    num_replacements_per_caption: int = 1,
    min_words_for_replacement: int = 3
) -> str:
    words = caption.split()
    if len(words) < min_words_for_replacement:
        return caption

    words_to_replace_indices = []
    for i, word in enumerate(words):
        if word.lower() in synonym_map:
            words_to_replace_indices.append(i)

    random.shuffle(words_to_replace_indices)

    new_words = list(words)
    replacements_made = 0

    for idx in words_to_replace_indices:
        if replacements_made >= num_replacements_per_caption:
            break

        original_word = new_words[idx].lower()
        if original_word in synonym_map:
            synonyms = synonym_map[original_word]
            if synonyms:
                chosen_synonym = random.choice(synonyms)
                if new_words[idx][0].isupper():
                    new_words[idx] = chosen_synonym.title()
                else:
                    new_words[idx] = chosen_synonym
                replacements_made += 1
    
    return " ".join(new_words)


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented text captions for training data."
    )
    parser.add_argument(
        "--train_original_csv",
        type=str,
        default=TRAIN_ORIGINAL_CSV,
        help="Path to the original training CSV (output from train/val split)."
    )
    parser.add_argument(
        "--train_augmented_csv",
        type=str,
        default=TRAIN_AUGMENTED_CSV,
        help="Path to save the augmented training CSV."
    )
    parser.add_argument(
        "--augmentation_method",
        type=str,
        default="manual",
        choices=["manual", "llm"],
        help="Method for text augmentation: 'manual' (synonym map) or 'llm' (LLM paraphrasing)."
    )
    parser.add_argument(
        "--num_aug_per_original",
        type=int,
        default=3,
        help="Number of augmented captions to generate per original caption (including original)."
    )
    parser.add_argument(
        "--manual_map_file",
        type=str,
        default=MANUAL_SYNONYM_MAP_FILE,
        help="Path to the JSON file for manual synonym mapping."
    )
    parser.add_argument(
        "--caption_key",
        type=str,
        default="caption",
        help="Column name for captions in the CSV."
    )
    parser.add_argument(
        "--img_key",
        type=str,
        default="filepath",
        help="Column name for image filepaths in the CSV."
    )
    
    args = parser.parse_args()

    # Load the manual synonym map (if it exists) or use the default
    manual_synonym_map = load_synonym_map(args.manual_map_file)
    if not manual_synonym_map:
        manual_synonym_map = DEFAULT_MANUAL_SYNONYM_MAP
        save_synonym_map(manual_synonym_map, args.manual_map_file)

    # Initialize LLM paraphraser if needed
    llm_paraphraser = None
    if args.augmentation_method == "llm":
        try:
            llm_paraphraser = LLMParaphraser(
                model_repo_id=LLM_MODEL_REPO_ID,
                model_filename=LLM_MODEL_FILENAME,
                local_path=LLM_MODEL_LOCAL_PATH
            )
        except ImportError as e:
            print(f"Error initializing LLM: {e}")
            print("Falling back to manual augmentation as LLM dependencies are not met.")
            args.augmentation_method = "manual"


    # Load original training data
    if not os.path.exists(args.train_original_csv):
        print(f"Error: {args.train_original_csv} not found. Please run the initial data split first.")
        exit(1)
    train_original_df = pd.read_csv(args.train_original_csv)

    augmented_data = []

    print(f"Starting text augmentation using '{args.augmentation_method}' method...")

    for index, row in train_original_df.iterrows():
        filepath = row[args.img_key]
        original_caption = row[args.caption_key]
        
        # Always include the original caption
        augmented_data.append({args.img_key: filepath, args.caption_key: original_caption})

        num_new_variations = args.num_aug_per_original - 1
        if num_new_variations <= 0:
            continue # Only add original if num_aug_per_original is 1 or less

        if args.augmentation_method == "manual":
            for _ in range(num_new_variations):
                augmented_caption = apply_manual_synonym_augmentation(
                    original_caption,
                    manual_synonym_map,
                    num_replacements_per_caption=random.randint(1, min(2, original_caption.count(' '))) # Randomly replace 1 or 2 words, or fewer if caption is short
                )
                if augmented_caption != original_caption: # Only add if actual change happened
                    augmented_data.append({args.img_key: filepath, args.caption_key: augmented_caption})
                else: # Fallback if no replacement happened (e.g., no matching words)
                    augmented_data.append({args.img_key: filepath, args.caption_key: f"A depiction of {original_caption.lower()}"})


        elif args.augmentation_method == "llm" and llm_paraphraser:
            llm_paraphrases = llm_paraphraser.paraphrase(original_caption, num_new_variations)
            for paraphrase in llm_paraphrases:
                augmented_data.append({args.img_key: filepath, args.caption_key: paraphrase})
    
    # Create new DataFrame from augmented data
    train_augmented_df = pd.DataFrame(augmented_data)

    # Shuffle the augmented data so that repeated images aren't consecutive in a batch
    train_augmented_df = train_augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the augmented training CSV
    train_augmented_df.to_csv(args.train_augmented_csv, index=False)
    print(f"Text augmentation complete. Saved augmented train data to {args.train_augmented_csv} with {len(train_augmented_df)} entries.")

if __name__ == "__main__":
    main()