import json
import os

def split_datafile(path_to_split, path_to_save):
    """
    Reads JSON files from a source directory and splits them into individual 
    files, one per item, using the 'qid' as part of the filename.
    """
    
    # Ensure the destination directory exists
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for file_name in os.listdir(path_to_split):
        file_path = os.path.join(path_to_split, file_name)
        
        # Only process files
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        # Iterate through every entry in the JSON list
        for item in data:
            item_id = item["qid"]
            
            # Construct a new unique filename (e.g., originalName_123)
            new_filename = f"{file_name}_{item_id}"
            save_path = os.path.join(path_to_save, new_filename)

            # Save the item wrapped in a list to match the original structure
            with open(save_path, "w") as fw:
                json.dump([item], fw)

if __name__ == "__main__":
    path_to_split = "my_path_to_split"
    path_to_save = "my_path_splitted"

    split_datafile(path_to_split, path_to_save)