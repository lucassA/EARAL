import json
import random
import torch
import torch.nn as nn
import numpy as np
import datasets
from functools import partial
from torch.utils.data import DataLoader
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments

random.seed(666)

# Global constants for identifier tokens
CHARACTERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"]
ACTUAL_CHARACTER_TOKENS = [" " + x for x in CHARACTERS]

# --- Custom Trainer Logic ---

class CustomBatchingTrainer(Trainer):
    """
    Custom Trainer that implements a dual loss: 
    1. Standard Cross-Entropy for language modeling.
    2. MSE Loss specifically for 'identifier tokens' to align probabilities.
    """
    def __init__(self, train_dataloader=None, eval_dataloader=None, identifier_tokens_ids=None, 
                 token_collator=None, token_padding=None, len_tok=None, alpha=0.5, *args, **kwargs):
        super(CustomBatchingTrainer, self).__init__(*args, **kwargs)

        self.token_collator = token_collator
        self.token_padding = token_padding
        self.alpha = alpha

        self.first_loss_function = nn.CrossEntropyLoss()
        self.second_loss_function = nn.MSELoss()

        self.train_dataloader_custom = train_dataloader
        self.eval_dataloader_custom = eval_dataloader
        self.identifier_tokens_ids = identifier_tokens_ids
        self.lentok = len_tok

    def get_train_dataloader(self):
        return self.train_dataloader_custom if self.train_dataloader_custom else super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader_custom if self.eval_dataloader_custom else super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels").to("cuda")
        id_labels = inputs.get("id_labels").to("cuda")

        # Masking padding for loss calculation
        attention_masks_bool = (labels != self.token_padding).to("cuda").bool()

        outputs = model.forward(input_ids)
        logits = outputs.get("logits")
        
        # Standard LM loss on masked tokens
        logits_masked = logits[attention_masks_bool]
        labels_masked = labels[attention_masks_bool]
        masked_loss = self.first_loss_function(logits_masked, labels_masked)

        # Identifier specific loss (MSE)
        # We extract only the logits corresponding to our specific characters
        logits_id_only = logits[:, :, self.identifier_tokens_ids]
        id_labels_float = id_labels.float()

        id_labels_masked = id_labels_float[attention_masks_bool]
        logits_id_only_masked = logits_id_only[attention_masks_bool]
        
        second_loss = self.second_loss_function(logits_id_only_masked, id_labels_masked)

        # Weighted combination of both losses
        final_loss = (self.alpha * masked_loss) + ((1 - self.alpha) * second_loss)

        return (final_loss, outputs) if return_outputs else final_loss

# --- Data Handling & Collation ---

def custom_collate_fn_(batch, id_pad_token):
    """
    Handles padding for a batch containing input_ids, token labels, 
    and vector-based id_labels (MSE targets).
    """
    max_seq_length = 2500
    input_prompts = [item['prompts'] for item in batch]
    preprompts = [item['preprompts'] for item in batch]
    labels = [item['labels'] for item in batch]
    id_labels = [item['id_labels'] for item in batch]

    # Determine batch padding length
    len_max_prompt = min(max(len(p) for p in input_prompts), max_seq_length)

    padded_prompt, padded_labels, padded_id_labels = [], [], []

    for fullprompt, preprompt, lbl, id_lbl in zip(input_prompts, preprompts, labels, id_labels):
        # 1. Pad input_ids (Right padding)
        pad_len = len_max_prompt - len(fullprompt)
        if pad_len > 0:
            fullprompt = fullprompt + ([id_pad_token] * pad_len)
        else:
            fullprompt = fullprompt[:len_max_prompt]

        # 2. Prepare Labels: we don't calculate loss on the preprompt (Instruction/Docs)
        # We pad the beginning of labels with the pad token so loss is ignored there
        len_preprompt = len(preprompt)
        lbl = ([id_pad_token] * (len_preprompt - 1)) + lbl
        id_lbl = ([[0] * len(CHARACTERS)] * (len_preprompt - 1)) + id_lbl

        # 3. Final alignment of labels to match len_max_prompt
        final_pad = len_max_prompt - len(lbl)
        if final_pad > 0:
            lbl = lbl + ([id_pad_token] * final_pad)
            id_lbl = id_lbl + ([[0] * len(CHARACTERS)] * final_pad)
        else:
            lbl = lbl[:len_max_prompt]
            id_lbl = id_lbl[:len_max_prompt]

        padded_prompt.append(fullprompt)
        padded_labels.append(lbl)
        padded_id_labels.append(id_lbl)

    return {
        'input_ids': torch.tensor(padded_prompt),
        'labels': torch.tensor(padded_labels),
        'id_labels': torch.tensor(padded_id_labels)
    }

# --- Main Training Entry Point ---

def start_training_fft(model_name_or_path, output_dir, logging_dir, nepoch, eos_token, assistant_end_token, 
                       path_dataset, path_eval_dataset, training_batch_size, gradient_acc, 
                       attn_implementation, marking_strategy, alpha):
    
    # Construct output paths
    suffix = f"mymodel_{marking_strategy}_alpha{alpha}"
    output_dir = f"{output_dir.rstrip('/')}/{suffix}"
    logging_dir = f"{logging_dir.rstrip('/')}/{suffix}"

    # Model configuration
    model_kwargs = {
        "torch_dtype": "auto",
        "use_cache": False,
        "device_map": "auto",
    }
    if attn_implementation == "flash_attention_2":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.gradient_checkpointing_enable()

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    id_pad_token = tokenizer.encode(tokenizer.pad_token)[-1]

    # Training configuration
    training_args = TrainingArguments(
        bf16=True,
        per_device_train_batch_size=training_batch_size,
        gradient_accumulation_steps=gradient_acc,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2.0e-05,
        log_level="info",
        logging_steps=25,
        logging_strategy="steps",
        lr_scheduler_type="constant",
        num_train_epochs=nepoch,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=1, 
        report_to="none",
        save_strategy="epochs",
        seed=42,
    )

    # Prepare specific token IDs for the custom loss
    identifier_tokens_ids = torch.tensor(
        tokenizer.encode(ACTUAL_CHARACTER_TOKENS, is_split_into_words=True, add_special_tokens=False), 
        device="cuda"
    )
    token_collator = tokenizer.encode(assistant_end_token)[-1]

    # Load and prepare dataset
    data_path = f"{path_dataset.rstrip('/')}/{marking_strategy}_train.json"
    with open(data_path, "r") as f:
        train_data = json.load(f)
    
    train_dataset = datasets.Dataset.from_list(train_data)

    # Standard DataLoader with our custom vector-padding collator
    custom_collate = partial(custom_collate_fn_, id_pad_token=id_pad_token)
    dataloader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        collate_fn=custom_collate,
        shuffle=True # The original code shuffled batches manually, shuffle=True achieves the same
    )

    trainer = CustomBatchingTrainer(
        model=model,
        args=training_args,
        train_dataloader=dataloader,
        identifier_tokens_ids=identifier_tokens_ids,
        token_collator=token_collator,
        token_padding=id_pad_token,
        len_tok=len(tokenizer),
        alpha=alpha,
    )

    trainer.train()