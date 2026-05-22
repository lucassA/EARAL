from transformers import AutoTokenizer
import datasets
import random
import re
import json
from utils import (
    add_docs, find_query_in_prompt, mark_texts, prepare_prompt, 
    find_docs_in_prompt, clean_evidence, compute_labels_batched_scale
)

random.seed(5)

# Shared prompt template used across all preprocessing functions
PROMPT_TEMPLATE = """Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided labelled search results (some of which might be irrelevant). 
Use an unbiased and journalistic tone. Always answer with knowledge only present in the labelled search results.
If none of the provided documents contain the answer, only respond with \"I apologize, but I couldn't find an answer to your question in the search results.\".

Question: {query}

Search results: {documents}

Answer:"""

# --- Entry Point Functions ---

def create_train_dataset(model_name_or_path, path_dataset, path_new_dataset, nb_docs_per_query, marking_strategy, characters=["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    process_datasets(tokenizer, path_dataset, path_new_dataset, characters, nb_docs_per_query, marking_strategy)

def create_eval_dataset(model_name_or_path, path_dataset, path_new_dataset, nb_docs_per_query, marking_strategy, characters=["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    process_eval_datasets(tokenizer, path_dataset, path_new_dataset, characters, nb_docs_per_query, marking_strategy)

# --- High Level Processors ---

def process_eval_datasets(tokenizer, path_dataset, path_new_dataset, CHARACTERS, nb_docs_per_query, marking_strategy):
    """Processes QAMPARI, ELI5, and ASQA datasets for evaluation."""
    with open(path_dataset + "/qampari_eval_gtr_top100.json", "r") as f:
        qampari = json.load(f)
    with open(path_dataset + "/eli5_eval_bm25_top100.json", "r") as f:
        eli5 = json.load(f)
    with open(path_dataset + "/asqa_eval_gtr_top100.json", "r") as f:
        asqa = json.load(f)

    asqa_ds = preprocess_dataset_asqa(tokenizer, asqa, CHARACTERS, nb_docs_per_query, marking_strategy)
    qampari_ds = preprocess_dataset_qampari(tokenizer, qampari, CHARACTERS, nb_docs_per_query, marking_strategy)
    eli5_ds = preprocess_dataset_eli5(tokenizer, eli5, CHARACTERS, nb_docs_per_query, marking_strategy)

    if path_new_dataset != "":
        output_map = {"QAMPARI": qampari_ds, "ELI5": eli5_ds, "ASQA": asqa_ds}
        for name, ds in output_map.items():
            with open(f"{path_new_dataset}/{name}/{marking_strategy}_eval.json", 'w') as f:
                json.dump(ds, f)
    return 0

def process_datasets(tokenizer, path_dataset, path_new_dataset, CHARACTERS, nb_docs_per_query, marking_strategy):
    """Merges HAGRID, Trust-Align, and ExpertsQA into a single training dataset."""
    hagrid_train = datasets.load_dataset(path_dataset + "/hagrid", split="train")
    hagrid_test = datasets.load_dataset(path_dataset + "/hagrid", split="dev")
    hagrid_all = datasets.concatenate_datasets([hagrid_train, hagrid_test])

    with open(path_dataset + "/trust-align/train.json", "r") as f:
        trust_align_train = [json.loads(line) for line in f]
    with open(path_dataset + "/expertsQA/r2_compiled_anon.jsonl", "r") as f:
        expertsQA_train = [json.loads(line) for line in f]
    
    # Collect all available documents for random negative sampling
    all_docs = []
    for queries in hagrid_all:
        for quote in queries['quotes']:
            all_docs.append((quote['docid'], quote['text']))
    for entry in trust_align_train:
        all_docs.extend(find_docs_in_prompt(entry["prompt"]))
    all_docs.extend(compute_alldocs_expertsQA(expertsQA_train))

    # Preprocess each source
    hagrid_ds = preprocess_dataset_hagrid(tokenizer, hagrid_all, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy)
    trust_ds = preprocess_dataset_trust_align(tokenizer, trust_align_train, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy)
    experts_ds = preprocess_dataset_expertdQA(tokenizer, expertsQA_train, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy)

    combined = datasets.concatenate_datasets([hagrid_ds, trust_ds, experts_ds])
    shuffled = combined.shuffle(seed=2)

    if path_new_dataset != "":
        data_list = [shuffled[i] for i in range(len(shuffled))]
        with open(f"{path_new_dataset}/{marking_strategy}_train.json", 'w') as f:
            json.dump(data_list, f)
    return shuffled

# --- Dataset-Specific Preprocessors ---

def preprocess_dataset_qampari(tokenizer, dataset, CHARACTERS, nb_docs_per_query, marking_strategy):
    dataset_items = []
    for entry in dataset:
        docs_for_query = []
        docs_raw = []
        for i in range(nb_docs_per_query):
            docs_for_query.append((entry["docs"][i]["id"], entry["docs"][i]["title"] + " " + entry["docs"][i]["text"]))
            docs_raw.append(entry["docs"][i])

        if len(docs_for_query) > len(CHARACTERS):
            continue

        new_docs, _ = mark_texts(docs_for_query, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
        preprompt = prepare_prompt(tokenizer, query=entry["question"], docs=new_docs, prompt_template=PROMPT_TEMPLATE)
        
        entry["qid"] = entry["id"]
        entry["query_text"] = entry["question"]
        entry["docs"] = docs_raw
        for i in range(nb_docs_per_query):
            entry["docs"][i]["identifier_tk"] = new_docs[i][1]
            entry["docs"][i]["markeddoc_text"] = new_docs[i][2]
        entry["prompt"] = tokenizer(preprompt, add_special_tokens=False).get("input_ids")
        dataset_items.append(entry)
    return dataset_items

def preprocess_dataset_eli5(tokenizer, dataset, CHARACTERS, nb_docs_per_query, marking_strategy):
    dataset_items = []
    for i, entry in enumerate(dataset):
        docs_for_query = []
        docs_raw = []
        for j in range(nb_docs_per_query):
            docs_for_query.append(("", entry["docs"][j]["title"] + " " + entry["docs"][j]["text"]))
            docs_raw.append(entry["docs"][j])

        if len(docs_for_query) > len(CHARACTERS):
            continue

        new_docs, _ = mark_texts(docs_for_query, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
        preprompt = prepare_prompt(tokenizer, query=entry["question"], docs=new_docs, prompt_template=PROMPT_TEMPLATE)

        entry["qid"] = i
        entry["query_text"] = entry["question"]
        entry["docs"] = docs_raw
        for j in range(nb_docs_per_query):
            entry["docs"][j]["identifier_tk"] = new_docs[j][1]
            entry["docs"][j]["markeddoc_text"] = new_docs[j][2]
        entry["prompt"] = tokenizer(preprompt, add_special_tokens=False).get("input_ids")
        dataset_items.append(entry)
    return dataset_items

def preprocess_dataset_asqa(tokenizer, dataset, CHARACTERS, nb_docs_per_query, marking_strategy):
    dataset_items = []
    for entry in dataset:
        docs_for_query = []
        docs_raw = []
        for i in range(nb_docs_per_query):
            docs_for_query.append((entry["docs"][i]["id"], entry["docs"][i]["title"] + " " + entry["docs"][i]["text"]))
            docs_raw.append(entry["docs"][i])

        if len(docs_for_query) > len(CHARACTERS):
            continue
        
        new_docs, _ = mark_texts(docs_for_query, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
        preprompt = prepare_prompt(tokenizer, query=entry["question"], docs=new_docs, prompt_template=PROMPT_TEMPLATE)

        entry["qid"] = entry["sample_id"]
        entry["query_text"] = entry["question"]
        entry["docs"] = docs_raw
        for i in range(nb_docs_per_query):
            entry["docs"][i]["identifier_tk"] = new_docs[i][1]
            entry["docs"][i]["markeddoc_text"] = new_docs[i][2]
        entry["prompt"] = tokenizer(preprompt, add_special_tokens=False).get("input_ids")
        dataset_items.append(entry)
    return dataset_items

def preprocess_dataset_hagrid(tokenizer, dataset, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy):
    all_prompts, all_preprompts, all_labels_tk, all_labels_ids = [], [], [], []

    for queries in dataset:
        query_text = queries['query']
        docs = [(q['docid'], q['text']) for q in queries['quotes']]
        len_initial_doc = len(docs)

        if len(docs) > len(CHARACTERS):
            continue

        for answer in queries['answers']:
            answer_text = answer['answer']
            # Remove citations and standardize 'search results' terminology
            answer_clean = re.sub(r'\[[0-9].*?\]', '', answer_text)
            answer_clean = answer_clean.replace('contexts', 'search results').replace('context', 'search results')

            sentences = answer['sentences']
            for sent in sentences:
                sent["text"] = sent["text"].replace('contexts', 'search results').replace('context', 'search results')

            if answer['attributable'] == 1:
                # Add distraction docs, mark indices, and compute token labels
                docs_filled = add_docs(docs, nb_docs_per_query, all_docs)
                new_docs, shuffled_indices = mark_texts(docs_filled, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
                tk_labels, id_labels = compute_labels_batched_scale(tokenizer, sentences, new_docs, shuffled_indices=shuffled_indices, len_initial_doc=len_initial_doc, CHARACTERS=CHARACTERS, eos_token=tokenizer.eos_token, scale="scaled")

                random.shuffle(new_docs) # Shuffle prompt order
                preprompt = prepare_prompt(tokenizer, query=query_text, docs=new_docs, prompt_template=PROMPT_TEMPLATE)
                
                all_preprompts.append(tokenizer(preprompt, add_special_tokens=False).get("input_ids"))
                all_prompts.append(tokenizer(preprompt + answer_clean, add_special_tokens=False).get("input_ids"))
                all_labels_tk.append(tk_labels)
                all_labels_ids.append(id_labels)

                # Data Augmentation: 25% chance to add a negative response if no answer is found
                if random.randint(1, 4) == 4:
                    neg_text = "I apologize, but I couldn't find an answer to your question in the search results."
                    neg_docs = add_docs([], nb_docs_per_query, all_docs)
                    new_docs_neg, shuffled_neg = mark_texts(neg_docs, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
                    tk_neg, id_neg = compute_labels_batched_scale(tokenizer, [{"attributable": 0, "text": neg_text}], new_docs_neg, shuffled_indices=shuffled_neg, len_initial_doc=0, CHARACTERS=CHARACTERS, eos_token=tokenizer.eos_token, scale="scaled")

                    random.shuffle(new_docs_neg)
                    pre_neg = prepare_prompt(tokenizer, query=query_text, docs=new_docs_neg, prompt_template=PROMPT_TEMPLATE)
                    all_preprompts.append(tokenizer(pre_neg, add_special_tokens=False).get("input_ids"))
                    all_prompts.append(tokenizer(pre_neg + neg_text, add_special_tokens=False).get("input_ids"))
                    all_labels_tk.append(tk_neg)
                    all_labels_ids.append(id_neg)

    dataset_items = [{"prompts": p, "preprompts": pp, "labels": l, "id_labels": il} for p, pp, l, il in zip(all_prompts, all_preprompts, all_labels_tk, all_labels_ids)]
    return datasets.Dataset.from_list(dataset_items)

def preprocess_dataset_trust_align(tokenizer, dataset, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy):
    all_prompts, all_preprompts, all_labels_tk, all_labels_ids = [], [], [], []

    for entry in dataset:
        answer = entry["chosen"]
        sentences = [{"attributable": 0 if "I apologize," in answer else 1, "text": answer}]
        docs = find_docs_in_prompt(entry["prompt"])
        query_text = find_query_in_prompt(entry["prompt"])

        if len(docs) > len(CHARACTERS):
            continue

        len_initial_doc = len(docs)
        docs_filled = add_docs(docs, nb_docs_per_query, all_docs)
        new_docs, shuffled_indices = mark_texts(docs_filled, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
        tk_labels, id_labels = compute_labels_batched_scale(tokenizer, sentences, new_docs, shuffled_indices=shuffled_indices, len_initial_doc=len_initial_doc, CHARACTERS=CHARACTERS, eos_token=tokenizer.eos_token, scale="scaled")

        random.shuffle(new_docs)
        preprompt = prepare_prompt(tokenizer, query=query_text, docs=new_docs, prompt_template=PROMPT_TEMPLATE)
        answer_clean = re.sub(r'\[[0-9].*?\]', '', answer)

        all_preprompts.append(tokenizer(preprompt, add_special_tokens=False).get("input_ids"))
        all_prompts.append(tokenizer(preprompt + answer_clean, add_special_tokens=False).get("input_ids"))
        all_labels_tk.append(tk_labels)
        all_labels_ids.append(id_labels)

    dataset_items = [{"prompts": p, "preprompts": pp, "labels": l, "id_labels": il} for p, pp, l, il in zip(all_prompts, all_preprompts, all_labels_tk, all_labels_ids)]
    return datasets.Dataset.from_list(dataset_items)

# --- ExpertsQA Helpers ---

def compute_alldocs_expertsQA(dataset):
    """Aggregates all cleaned evidence sentences from ExpertsQA entries."""
    all_docs = {}
    new_all_docs = []
    for entry in dataset:
        question = entry["question"]
        if question not in all_docs:
            all_docs[question] = {}
        for ans_key in entry["answers"]:
            for claim in entry["answers"][ans_key]["claims"]:
                evidences = claim["evidence"] if isinstance(claim["evidence"], list) else [claim["evidence"]]
                for evi in evidences:
                    clean_txt, num = clean_evidence(evi)
                    if clean_txt:
                        if num not in all_docs[question]:
                            all_docs[question][num] = [clean_txt]
                        else:
                            for part in clean_txt.split('.'):
                                if not any(part in existing for existing in all_docs[question][num]):
                                    all_docs[question][num].append(part)
                
                # Check revised evidence if available
                if "revised_evidence" in claim and claim["revised_evidence"]:
                    rev_evi = claim["revised_evidence"]
                    rev_evis = rev_evi if isinstance(rev_evi, list) else [rev_evi]
                    for evi in rev_evis:
                        if not evi: continue
                        clean_txt, num = clean_evidence(evi)
                        if clean_txt:
                            if num not in all_docs[question]:
                                all_docs[question][num] = [clean_txt]
                            else:
                                for part in clean_txt.split('.'):
                                    if not any(part in existing for existing in all_docs[question][num]):
                                        all_docs[question][num].append(part)

    for question in all_docs:
        for num in all_docs[question]:
            concat_str = " ".join(all_docs[question][num])
            if len(re.findall("[A-Za-z]", concat_str)) > 1:
                new_all_docs.append(concat_str)
    return new_all_docs

def preprocess_dataset_expertdQA(tokenizer, dataset, all_docs, CHARACTERS, nb_docs_per_query, marking_strategy):
    """Handles complex citation mapping and revised claims for ExpertsQA."""
    dataaugm_expertsQA = True
    all_evidence_texts = {}

    # Gather evidence per question
    for entry in dataset:
        question = entry["question"]
        if question not in all_evidence_texts:
            all_evidence_texts[question] = {}
        for ans_key in entry["answers"]:
            for claim in entry["answers"][ans_key]["claims"]:
                evis = claim["evidence"] if isinstance(claim["evidence"], list) else [claim["evidence"]]
                for evi in evis:
                    txt, num = clean_evidence(evi)
                    if txt:
                        if num not in all_evidence_texts[question]:
                            all_evidence_texts[question][num] = [txt]
                        else:
                            for part in txt.split('.'):
                                if not any(part in existing for existing in all_evidence_texts[question][num]):
                                    all_evidence_texts[question][num].append(part)

    # Remap citation indices for the specific question
    new_all_evidence_texts = {}
    all_switch = {}
    for question in all_evidence_texts:
        all_switch[question] = {}
        new_all_evidence_texts[question] = {}
        idx = 0
        for num in all_evidence_texts[question]:
            concat_str = " ".join(all_evidence_texts[question][num])
            if len(re.findall("[A-Za-z]", concat_str)) > 1:
                new_all_evidence_texts[question][idx + 1] = concat_str
                all_switch[question][num] = idx + 1
            else:
                idx -= 1
            idx += 1

    all_prompts, all_preprompts, all_labels_tk, all_labels_ids = [], [], [], []

    for entry in dataset:
        question = entry["question"]
        for ans_key in entry["answers"]:
            claims = entry["answers"][ans_key]["claims"]
            docs = [(n, new_all_evidence_texts[question][n]) for n in new_all_evidence_texts[question]]
            len_initial_doc = len(docs)
           
            if len(docs) > len(CHARACTERS):
                continue

            answer_text = ""
            sentences = []
            
            # Extract claims, handling revised versions and citation splitting
            for claim in claims:
                if "revised_claim" in claim and "revised_evidence" in claim:
                    revised_claim = claim["revised_claim"]
                    revised_evidence = claim["revised_evidence"]
                    if not revised_claim or not revised_evidence:
                        claim_text, attributable = "", 0
                    else:
                        claim_text, attributable = revised_claim, 1
                        citations = list(dict.fromkeys(re.findall("\[[0-9]\]", claim_text)))

                        if len(citations) == 0:
                            new_cits = list(dict.fromkeys(re.findall("\[[0-9]\]", revised_evidence)))
                            for cit in new_cits:
                                claim_text += cit
                            if claim_text:
                                answer_text = (answer_text + ' ' + claim_text) if answer_text else claim_text
                                sentences.append({"attributable": attributable, "text": claim_text})
                        elif len(citations) == 1:
                            if claim_text:
                                answer_text = (answer_text + ' ' + claim_text) if answer_text else claim_text
                                sentences.append({"attributable": attributable, "text": claim_text})
                        else:
                            splits = [s for s in re.split("\[[0-9]\]", claim_text) if s.strip() not in ["", ".", ",", ", "]]
                            if len(splits) != len(citations):
                                sentences.append({"attributable": 1, "text": claim_text})
                            else:
                                for i in range(len(splits)):
                                    sentences.append({"attributable": 1, "text": f"{splits[i]} {citations[i]}"})
                else:
                    claim_text, attributable = claim["claim_string"], 0
                    if claim["correctness"] and claim["support"] == "Complete" and " correct" in claim["correctness"]:
                        attributable = 1
                    else:
                        claim_text = ""

                    if claim_text:
                        answer_text = (answer_text + ' ' + claim_text) if answer_text else claim_text
                        sentences.append({"attributable": attributable, "text": claim_text})

            if not answer_text:
                answer_text = "I apologize, but I couldn't find an answer to your question in the search results."
                sentences = [{"attributable": 0, "text": answer_text}]

            # Map citation numbers back to internal indices
            new_sentences = []
            at_least_one_cited = False
            for sent in sentences:
                if sent["attributable"] == 1:
                    cits = list(dict.fromkeys(re.findall("\[[0-9]\]", sent["text"])))
                    txt_no_cit = re.sub("\[[0-9]\]", "", sent["text"])
                    cited = False
                    for cit in cits:
                        num = re.findall("[0-9]", cit)[0]
                        if num in all_switch[question]:
                            txt_no_cit += f"[{all_switch[question][num]}]"
                            cited = True
                            at_least_one_cited = True
                    if cited:
                        txt_no_cit = txt_no_cit.replace(" .", ".").strip()
                        new_sentences.append({"attributable": 1, "text": txt_no_cit})
                else:
                    sent["text"] = sent["text"].strip()
                    new_sentences.append(sent)

            if not at_least_one_cited:
                answer_text = "I apologize, but I couldn't find an answer to your question in the search results."
                new_sentences = [{"attributable": 0, "text": answer_text}]
            
            # Clean up sentence fragments/whitespace
            i = len(new_sentences) - 1
            while i > 0:
                if re.findall("\[[0-9]\]", new_sentences[i]["text"]):
                    txt_clean = re.sub("\[[0-9]\]", "", new_sentences[i]["text"])
                    if not txt_clean.strip():
                        new_sentences[i-1]["text"] += new_sentences[i]["text"]
                i -= 1

            final_sentences = []
            for sent in new_sentences:
                if re.sub("\[[0-9]\]", "", sent["text"]).strip():
                    sent["text"] = sent["text"].strip().replace('\\n', '')
                    final_sentences.append(sent)

            # Rebuild clean answer for training
            answer_clean = ""
            for sent in final_sentences:
                txt_clean = re.sub(r'\[[0-9].*?\]', '', sent["text"])
                if not answer_clean:
                    answer_clean = txt_clean
                else:
                    answer_clean += txt_clean if txt_clean.startswith(" ") else " " + txt_clean

            # Prepare final prompt/labels
            docs_filled = add_docs(docs, nb_docs_per_query, all_docs)
            new_docs, shuffled_indices = mark_texts(docs_filled, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
            tk_labels, id_labels = compute_labels_batched_scale(tokenizer, final_sentences, new_docs, shuffled_indices=shuffled_indices, len_initial_doc=len_initial_doc, CHARACTERS=CHARACTERS, eos_token=tokenizer.eos_token, scale="scaled")

            random.shuffle(new_docs)
            preprompt = prepare_prompt(tokenizer, query=question, docs=new_docs, prompt_template=PROMPT_TEMPLATE)
            
            all_preprompts.append(tokenizer(preprompt, add_special_tokens=False).get("input_ids"))
            all_prompts.append(tokenizer(preprompt + answer_clean, add_special_tokens=False).get("input_ids"))
            all_labels_tk.append(tk_labels)
            all_labels_ids.append(id_labels)

            # Optional data augmentation (negative samples)
            if dataaugm_expertsQA and random.randint(1, 4) == 4:
                neg_text = "I apologize, but I couldn't find an answer to your question in the search results."
                neg_docs = add_docs([], nb_docs_per_query, all_docs)
                new_docs_neg, shuffled_neg = mark_texts(neg_docs, marking_strategy=marking_strategy, CHARACTERS=CHARACTERS)
                tk_neg, id_neg = compute_labels_batched_scale(tokenizer, [{"attributable": 0, "text": neg_text}], new_docs_neg, shuffled_indices=shuffled_neg, len_initial_doc=0, CHARACTERS=CHARACTERS, eos_token=tokenizer.eos_token, scale="scaled")
                
                random.shuffle(new_docs_neg)
                pre_neg = prepare_prompt(tokenizer, query=question, docs=new_docs_neg, prompt_template=PROMPT_TEMPLATE)
                all_preprompts.append(tokenizer(pre_neg, add_special_tokens=False).get("input_ids"))
                all_prompts.append(tokenizer(pre_neg + neg_text, add_special_tokens=False).get("input_ids"))
                all_labels_tk.append(tk_neg)
                all_labels_ids.append(id_neg)

    dataset_items = [{"prompts": p, "preprompts": pp, "labels": l, "id_labels": il} for p, pp, l, il in zip(all_prompts, all_preprompts, all_labels_tk, all_labels_ids)]
    return datasets.Dataset.from_list(dataset_items)