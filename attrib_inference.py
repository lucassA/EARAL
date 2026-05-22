import torch
import random
import json
import os
import numpy as np
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial

random.seed(2)

# Global constants for identifier tokens
CHARACTERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"]
ACTUAL_CHARACTER_TOKENS = [" " + x for x in CHARACTERS]

class Inferencer():
    """
    Handles model generation and specifically extracts the probabilities/logits 
    for the character identifiers used in the dual-loss training.
    """
    def __init__(self, model, token_padding=None, identifier_tokens_ids=None, max_new_tokens=100):
        self.model = model
        self.token_padding = token_padding
        self.identifier_tokens_ids = identifier_tokens_ids
        self.max_new_tokens = max_new_tokens

    def run_query_inference_optimized(self, inputs):
        input_ids = inputs.get("input_ids").to("cuda")
        # For inference, padding is usually on the left, but mask must be correct
        attention_mask = (input_ids != self.token_padding).long().to("cuda")
        input_length = input_ids.shape[1]

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False, # Greedy decoding
            num_beams=1,                  
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_tokens = output.sequences[:, input_length:]

        # Extract logits for each generated step
        logits = torch.stack(output.scores, dim=1).to("cuda")  
        logits_softmax = torch.softmax(logits, dim=-1)
        
        # Prepare indices to gather only the identifier token probabilities
        # Shape: [batch, sequence_length, nb_identifiers]
        expanded_ids = torch.tensor(self.identifier_tokens_ids).unsqueeze(0).unsqueeze(0)
        expanded_ids = expanded_ids.expand(logits_softmax.size(0), logits_softmax.size(1), -1).to("cuda")

        probas_identifiers = torch.gather(logits_softmax, dim=-1, index=expanded_ids)
        logits_identifier = torch.gather(logits, dim=-1, index=expanded_ids)
    
        return generated_tokens, probas_identifiers, logits_identifier

def custom_collate_fn_(batch, id_pad_token):
    """
    Custom collator for inference. 
    Uses LEFT-PADDING because decoder-only models require it for generation.
    """
    grouped_batch = {key: [d[key] for d in batch] for key in batch[0]}
    prompts = [item['prompt'] for item in batch]
    len_max_prompt = max(len(p) for p in prompts)
    
    padded_prompts = []
    for prompt in prompts:
        # Left padding logic
        padding = [id_pad_token] * (len_max_prompt - len(prompt))
        padded_prompts.append(padding + prompt)

    grouped_batch["input_ids"] = torch.tensor(padded_prompts)
    return grouped_batch

def create_inference_dataloader_from_dataset_path(dataset_path, marking_strategy, batch_size, custom_collate):
    """Loads evaluation dataset and creates a DataLoader sorted by length for efficiency."""
    file_path = f"{dataset_path.rstrip('/')}/{marking_strategy}_evalv2.json"
    with open(file_path, "r") as f:
        eval_data = json.load(f)

    eval_dataset = datasets.Dataset.from_list(eval_data)
    
    # Sort by length to minimize padding in batches
    all_lengths = np.array([len(entry["prompt"]) for entry in eval_dataset])
    sorted_indices = all_lengths.argsort()[::-1]
    
    # Split into batch-sized chunks and shuffle the batches
    batches = np.array_split(sorted_indices, np.ceil(len(sorted_indices) / batch_size))
    random.shuffle(batches)
    
    return DataLoader(eval_dataset, collate_fn=custom_collate, batch_sampler=batches)

def init_and_perform_inference(model_name_or_path, path_trained_model, path_dataset, path_savecontribs, 
                               marking_strategy, eos_token, attn_implementation, inference_batch_size, 
                               alpha, checkpoint):
    """
    Initializes the model and runs inference across the three target datasets.
    """
    # Build the path to the specific checkpoint
    trained_model_subdir = f"Llama31fft_{marking_strategy}_alpha{alpha}"
    full_model_path = os.path.join(path_trained_model, trained_model_subdir)
    if checkpoint:
        full_model_path = os.path.join(full_model_path, checkpoint)

    if not os.path.exists(full_model_path):
        print(f"Path not found: {full_model_path}")
        return

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        full_model_path, 
        torch_dtype="auto",
        use_cache=False,
        device_map='auto',
        attn_implementation=attn_implementation,
    ).eval()

    # Load Tokenizer and special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    id_pad_token = tokenizer.encode(tokenizer.pad_token)[-1]    
    
    # Get IDs for the identifier characters (AA, BB, etc.)
    identifier_tokens_ids = tokenizer.encode(ACTUAL_CHARACTER_TOKENS, is_split_into_words=True, add_special_tokens=False)

    inferencer = Inferencer(model, token_padding=id_pad_token, identifier_tokens_ids=identifier_tokens_ids, max_new_tokens=150)
    custom_collate = partial(custom_collate_fn_, id_pad_token=id_pad_token)

    # Inference loop across datasets
    for ds_name in ["QAMPARI", "ELI5", "ASQA"]:
        ds_path = f"{path_dataset.rstrip('/')}/{ds_name}"
        inference_dataloader = create_inference_dataloader_from_dataset_path(ds_path, marking_strategy, inference_batch_size, custom_collate)
        
        all_results = []
        for batch in inference_dataloader:
            generated_tokens, probas_ids, logits_ids = inferencer.run_query_inference_optimized(batch)
            
            # Remove tensor from dict for processing
            batch.pop("input_ids")
            # Convert batch-of-lists back to list-of-dicts
            keys = batch.keys()
            ungrouped = [dict(zip(keys, values)) for values in zip(*batch.values())]
            
            for i in range(len(ungrouped)):
                ungrouped[i]["generated_tokens"] = generated_tokens[i].tolist()
                ungrouped[i]["generated_tokens_tk"] = tokenizer.convert_ids_to_tokens(generated_tokens[i])
                ungrouped[i]["p_ident"] = probas_ids[i].tolist()
                ungrouped[i]["l_ident"] = logits_ids[i].tolist()
                all_results.append(ungrouped[i])

        # Save results for current dataset
        output_filename = f"{ds_name}_{marking_strategy}_alpha{alpha}.json"
        save_path = os.path.join(path_savecontribs, output_filename)
        with open(save_path, 'w') as f:
            json.dump(all_results, f)

def perform_multiple_inf(model_name_or_path, path_trained_model, path_dataset, path_savecontribs, 
                         marking_strategy, eos_token, attn_implementation, inference_batch_size, 
                         alpha, checkpoint):
    """
    Wrapper to handle comma-separated strings for hyperparameter grid evaluation.
    """
    # Parse inputs
    strategies = marking_strategy.split(',') if ',' in marking_strategy else [marking_strategy]
    
    if ',' in str(alpha):
        alphas = [float(a) for a in alpha.split(',')]
    else:
        alphas = [float(alpha)]

    # Nested loops for hyperparameter sweep
    for mrk in strategies:
        for alp in alphas:
            init_and_perform_inference(
                model_name_or_path, path_trained_model, path_dataset, 
                path_savecontribs, mrk, eos_token, attn_implementation, 
                inference_batch_size, alp, checkpoint
            )