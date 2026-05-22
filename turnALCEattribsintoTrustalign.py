import os
import json

def process_data(asqa_file, eli5_file, qampari_file, folder_to_process, folder_to_save):
    # Load master datasets
    with open(asqa_file, 'r') as f:
        all_asqa = json.load(f)
    with open(eli5_file, 'r') as f:
        all_eli5 = json.load(f)
    with open(qampari_file, 'r') as f:
        all_qampari = json.load(f)

    # Convert lists to dictionaries for O(1) lookup by question
    # This avoids nested loops and significantly speeds up processing
    master_maps = {
        "ASQA": {item["question"]: item for item in all_asqa},
        "ELI5": {item["question"]: item for item in all_eli5},
        "QAMPARI": {item["question"]: item for item in all_qampari}
    }

    # Ensure save directory exists
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

    for file_name in os.listdir(folder_to_process):
        save_path = os.path.join(folder_to_save, file_name)
        
        # Skip if already processed
        if os.path.exists(save_path):
            continue

        # Determine which master dataset to use based on the filename
        active_map = None
        for key in master_maps:
            if key in file_name:
                active_map = master_maps[key]
                break
        
        if active_map is None:
            continue

        # Load and process the specific file
        file_path = os.path.join(folder_to_process, file_name)
        with open(file_path, 'r') as f:
            items = json.load(f)

        for item in items:
            question = item.get("question")
            master_item = active_map.get(question)

            if master_item:
                # Sync document metadata (limited to the first 5 docs)
                docs_item = item.get("docs", [])
                docs_master = master_item.get("docs", [])[:5]

                for doc_item, doc_master in zip(docs_item, docs_master):
                    doc_item["answers_found"] = doc_master.get("answers_found")
                    doc_item["rec_score"] = doc_master.get("rec_score")

                # Copy answers from master data
                item["answers"] = master_item.get("answers")
            else:
                print(f"Item not found for question: {question}")

        # Save the updated data
        with open(save_path, 'w') as f:
            json.dump(items, f)