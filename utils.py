import random
import re 

def add_docs(docs, nb_docs_per_query, all_docs):
    """
    Fills the document list with random samples from all_docs until 
    the target number of documents is reached.
    """
    if len(docs) < nb_docs_per_query:
        nb_completing_docs = random.randint(0, nb_docs_per_query - len(docs))

        for i in range(nb_completing_docs):
            new_doc = random.choice(all_docs)
            safeguard = 0
            # Ensure we don't pick a document already in the current set
            while new_doc in docs and safeguard < 50:
                new_doc = random.choice(all_docs)
                safeguard += 1
            
            if safeguard >= 50:
                print("exit because of safeguard during doc sampling")
            
            docs.append(new_doc)

    return docs

def mark_texts(texts, marking_strategy, CHARACTERS):
    """
    Applies different formatting strategies to documents using unique identifiers.
    Strategies include:
    - BA: Brackets around the whole text.
    - BAS: Brackets around every sentence.
    - AW: Identifier inserted before every word.
    - weak: Simple "Document [ID]:" prefix.
    """
    potential_indentifiers = CHARACTERS.copy() 
    new_texts = []

    # Randomly assign identifiers to documents
    indices = list(range(len(potential_indentifiers)))
    random.shuffle(indices)
    identifier_list = [potential_indentifiers[i] for i in indices]

    for text, id_char in zip(texts, identifier_list[:len(texts)]):
        new_text = ""
        identifier = id_char

        if marking_strategy == "BA":
            new_text = f"< {identifier}>{text[1]}</ {identifier}>"

        elif marking_strategy == "BAS":
            # Split by period and wrap every sentence in tags
            pattern = r'[.]'
            doc_split = [s.strip() for s in re.split(pattern, text[1]) if s.strip()]
            for doc_s in doc_split:
                tag = f"< {identifier}>{doc_s}.</ {identifier}>"
                new_text = tag if new_text == "" else f"{new_text} {tag}"
                    
        elif marking_strategy == "AW":
            # Insert identifier before every word
            doc_split = text[1].split()
            new_text = " ".join([f"{identifier} {word}" for word in doc_split])

        elif marking_strategy == "weak":
            new_text = f"Document [ {identifier}]: {text[1]}"

        new_texts.append((text[0], identifier, new_text))

    return new_texts, indices

def compute_labels_batched_scale(tokenizer, sentences, new_docs, shuffled_indices, len_initial_doc, CHARACTERS, eos_token, scale="normal"):
    """
    Generates token-level labels for the answer sentences. 
    Determines if a sentence is attributable to specific document IDs 
    based on the presence of citations.
    """
    all_labels_tk_this_answer = []
    all_labels_ids_this_answer = []
    allchosen_id = [nd[1] for nd in new_docs]

    for count, sentence in enumerate(sentences):
        attributable_sentence_score = sentence['attributable']
        sentence_text = sentence['text']
        all_attrib_identifiers_for_this_sentence = []

        # Determine attribution scores for each possible identifier
        for i in range(1, len(CHARACTERS) + 1):
            if CHARACTERS[shuffled_indices[i-1]] not in allchosen_id:
                all_attrib_identifiers_for_this_sentence.append(0 if scale == "normal" else -2)
            else:
                # Check if citation exists in text (e.g., [1], 1], etc.)
                citation_patterns = [f'[{i}]', f'{i}]', f'[{i}', f', {i},', f'context {i},']
                has_citation = any(p in sentence_text for p in citation_patterns)
                
                if attributable_sentence_score == 1 and has_citation:
                    all_attrib_identifiers_for_this_sentence.append(1 if scale == "normal" else 4)
                elif i <= len_initial_doc:
                    all_attrib_identifiers_for_this_sentence.append(0 if scale == "normal" else 2)
                else:
                    all_attrib_identifiers_for_this_sentence.append(0)

        # Map scores back to original index order
        all_attrib_identifiers_for_this_sentence_ = [0] * len(CHARACTERS)
        for j, idx in enumerate(shuffled_indices):
            all_attrib_identifiers_for_this_sentence_[idx] = all_attrib_identifiers_for_this_sentence[j]

        # Remove citations from text before tokenization
        sentence_wo_attrib = re.sub(r'\[[0-9].*?\]', '', sentence_text)
        if count > 0:
            if not (sentence_wo_attrib.startswith(" ") or sentence_wo_attrib.startswith("\\n")):
                sentence_wo_attrib = " " + sentence_wo_attrib

        tokenized_sentence = tokenizer.encode(sentence_wo_attrib, add_special_tokens=False)

        # Apply the attribution vector to every token in the sentence
        for tk in tokenized_sentence:
            all_labels_tk_this_answer.append(tk)
            all_labels_ids_this_answer.append(all_attrib_identifiers_for_this_sentence_)

    # Append EOS token with a null label
    all_labels_tk_this_answer.append(tokenizer.encode(eos_token)[-1])
    all_labels_ids_this_answer.append([-1] * len(CHARACTERS))

    return all_labels_tk_this_answer, all_labels_ids_this_answer

def prepare_prompt(tokenizer, query="", docs=[], prompt_template=""):
    """
    Constructs the final prompt string using the tokenizer's chat template.
    """
    documents_str = "\n\n".join([f'{doc[2]}' for doc in docs])
    conversation = [{"role": "user", "content": prompt_template.format(query=query, documents=documents_str)}]
    
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

def find_docs_in_prompt(prompt):
    """
    Parses a prompt string to extract and clean documents based on bracketed IDs.
    """
    prompt_split = prompt.split("Document")
    all_docs = []
    for txt in prompt_split[1:]:
        if "Answer:" in txt:
            txt = re.sub("Answer", "", txt)

        for i in range(1, 11):
            if f'[{i}]' in txt:
                # Clean up labels and whitespace
                txt_wo_label = re.sub(r'[^a-zA-Z0-9]+$', '', txt)
                txt_wo_label = re.sub(f'\\[{i}\\]', "", txt_wo_label).lstrip(' ')
                all_docs.append((i, txt_wo_label))
                break
    return all_docs

def find_docs_in_prompt_vanilla(prompt):
    """
    Extracts documents from a prompt without aggressive text cleaning.
    """
    prompt_split = prompt.split("Document")
    all_docs = []
    for txt in prompt_split[1:]:
        if "Answer:" in txt:
            txt = re.sub("Answer", "", txt)
        for i in range(1, 11):
            if f'[{i}]' in txt:
                all_docs.append((i, txt))
                break
    return all_docs
                
def find_query_in_prompt(prompt):
    """
    Extracts the question from the prompt string.
    """
    query = re.findall(r'Question(.*?)Document', prompt, re.DOTALL)
    if not query:
        print("query not found")
        return ""
    
    query_clean = re.sub(r'^[^a-zA-Z0-9]+', '', query[0]).rstrip('\n')
    return query_clean

def clean_evidence(revised_evidence):
    """
    Removes URLs and citation brackets from a string.
    """
    if revised_evidence:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        citations = re.findall(r"\[[0-9]\]", revised_evidence)
        
        if citations:
            doc_id = re.findall(r"[0-9]", citations[0])[0]
            txt_wo_label = re.sub(r'\[[0-9]\]', "", revised_evidence)
            txt_wo_label = url_pattern.sub('', txt_wo_label)
            return txt_wo_label, doc_id
            
    return "", ""