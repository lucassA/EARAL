import json
import os
import re
from transformers import AutoTokenizer
from nltk import sent_tokenize

# Global constants for identifier characters
CHARACTERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"]

def find_sentence_indices(generated_tokens, tokenizer):
    """
    Splits decoded text into sentences and finds the start/end token 
    indices for each sentence within the generated sequence.
    """
    decoded_text = tokenizer.decode(generated_tokens)
    raw_sentences = sent_tokenize(decoded_text)
    
    # Standardize sentence spacing (often needed for correct re-tokenization)
    sentences = []
    for i, s in enumerate(raw_sentences):
        if i > 0 and not s.startswith(" "):
            sentences.append(" " + s)
        else:
            sentences.append(s)

    sentence_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    
    indices_sent = []
    current_idx = 0
    for s_tok in sentence_tokens:
        indices_sent.append((current_idx, current_idx + len(s_tok)))
        current_idx += len(s_tok)

    final_indices = []
    sent_texts_with_attribs = []

    for start, end in indices_sent:
        # Ignore very short sentences (e.g., single punctuation)
        if (end - start) <= 1:
            continue
            
        # Ensure indices don't exceed actual token length
        safe_end = min(end, len(generated_tokens))
        if start >= len(generated_tokens):
            continue
            
        actual_tokens = [generated_tokens[x] for x in range(start, safe_end)]
        sent_text = tokenizer.decode(actual_tokens, skip_special_tokens=True)
        
        final_indices.append((start, safe_end))
        # Sentences are stored as (text, [list_of_doc_indices])
        sent_texts_with_attribs.append([sent_text, []])

    return final_indices, sent_texts_with_attribs

def perform_aggregs(p_ident, l_ident, indices, aggreg_method, threshold_attr, threshold_count, contrib_to_use):
    """
    Aggregates token scores into a sentence-level attribution decision.
    """
    scores = p_ident if contrib_to_use == "probas" else l_ident
    start, end = indices
    sent_len = end - start
    attributions = []

    # Aggregation Strategy: Maximum 
    # (Attributed if any token in the sentence exceeds the threshold)
    if aggreg_method == "max":
        for j in range(len(CHARACTERS)):
            if any(scores[i][j] > threshold_attr for i in range(start, end)):
                attributions.append(CHARACTERS[j])
        
    # Aggregation Strategy: Proportion 
    # (Attributed if >X% of tokens in the sentence exceed the threshold)
    elif aggreg_method == "prop":
        for j in range(len(CHARACTERS)):
            count = sum(1 for i in range(start, end) if scores[i][j] > threshold_attr)
            if count > (threshold_count * sent_len):
                attributions.append(CHARACTERS[j])

    # Aggregation Strategy: Average 
    # (Attributed if the mean score across the sentence exceeds the threshold)
    elif aggreg_method == "avg":
        for j in range(len(CHARACTERS)):
            avg_val = sum(scores[i][j] for i in range(start, end)) / sent_len
            if avg_val > threshold_attr:
                attributions.append(CHARACTERS[j])

    return attributions

def attribution_revert_from_identifier_to_docindex(docs, attribution):
    """Maps internal identifiers (AA, BB...) back to 1-based document indices."""
    id_map = {doc["identifier_tk"]: i + 1 for i, doc in enumerate(docs)}
    
    unique_indices = []
    for attr in attribution:
        idx = id_map.get(attr)
        if idx and idx not in unique_indices:
            unique_indices.append(idx)
    return sorted(unique_indices)

def start_aggreg_(model_name_or_path, path_savecontribs, file_contrib, aggreg_method, 
                  path_output, threshold_attr, threshold_count, contrib_to_use, invfailsafe):
    """
    Processes a single inference file and generates an attributed final answer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    attribs_results = []
    
    output_filename = f"{file_contrib}_aggreg{aggreg_method}_thresh{threshold_attr}_threshC{threshold_count}.json"
    save_path = os.path.join(path_output, output_filename)
    
    if os.path.exists(save_path):
        return

    input_path = os.path.join(path_savecontribs, file_contrib)
    with open(input_path, "r") as f:
        items = json.load(f)

    for item in items:
        sent_indices, sent_data = find_sentence_indices(item["generated_tokens"], tokenizer)
        
        any_attribution_found = False
        final_output_text = ""

        for indices, sent_info in zip(sent_indices, sent_data):
            # 1. Determine which character identifiers are attributed
            attr_ids = perform_aggregs(item["p_ident"], item["l_ident"], indices, aggreg_method, 
                                       threshold_attr, threshold_count, contrib_to_use)
            
            # 2. Map identifiers to document numbers (1, 2...)
            doc_indices = attribution_revert_from_identifier_to_docindex(item["docs"], attr_ids)
            sent_info[1] = doc_indices
            
            # 3. Build text with citations (e.g., "The sky is blue [1].")
            clean_sent = sent_info[0].rstrip(' .')
            if "I apologize" not in sent_info[0]:
                if doc_indices:
                    any_attribution_found = True
                    citations = "".join([f" [{idx}]" for idx in doc_indices])
                    clean_sent += citations
            
            final_output_text += clean_sent + ". "

        # Failsafe: if no citations found and flag enabled, return apology
        if invfailsafe and not any_attribution_found:
            final_output_text = "I apologize, but I could not find any answer to your question in the search results."

        item["output"] = final_output_text.strip()
        attribs_results.append(item)

    with open(save_path, "w") as f:
        json.dump(attribs_results, f)

def perform_multiple_aggreg(model_name_or_path, path_savecontribs, file_contrib, aggreg_method, 
                            path_output, threshold_attr, threshold_count, contrib_to_use, invfailsafe):
    """
    Parses complex hyperparameter strings to run multiple aggregation configurations.
    
    Example threshold_attr format: "0.1,0.2:0.5" 
    (meaning 0.1 and 0.2 for the first method, 0.5 for the second).
    """
    methods = aggreg_method.split(',') if ',' in aggreg_method else [aggreg_method]

    # Parse thresholds mapped to each method
    thresh_parts = threshold_attr.split(':')
    thresh_map = {}
    for i, part in enumerate(thresh_parts):
        method_key = methods[i]
        if ',' in part:
            thresh_map[method_key] = [float(x) for x in part.split(',')]
        else:
            thresh_map[method_key] = [float(part)]

    # Parse proportion counts
    counts = [float(x) for x in threshold_count.split(',')] if ',' in str(threshold_count) else [float(threshold_count)]

    # Nested loops for hyperparameter sweep
    for method in methods:
        if method == "prop":
            for c_att in counts:
                for t_att in thresh_map[method]:
                    start_aggreg_(model_name_or_path, path_savecontribs, file_contrib, method, 
                                  path_output, t_att, c_att, contrib_to_use, invfailsafe)
        else:
            for t_att in thresh_map[method]: 
                start_aggreg_(model_name_or_path, path_savecontribs, file_contrib, method, 
                              path_output, t_att, counts[0], contrib_to_use, invfailsafe)