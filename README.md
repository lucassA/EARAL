# EARAL : Efficient Attribution of Retrieval-Augmented Answer Generation using Logits

This framework provides a complete pipeline for training, running inference, and evaluating Large Language Models on document attribution using EARAL.

## 📌 Project Overview
The pipeline consists of:
1.  **Dataset Creation**: Formatting raw data into attribution-ready JSON formats.
2.  **Training**: Fine-Tuning using specialized loss functions.
3.  **Inference**: Extracting model contributions (logits or probabilities) per token.
4.  **Aggregation**: Converting raw token contributions into sentence-level document attributions.
5.  **Conversion**: Formatting results for the **Trust-Align** evaluation suite.

---

## 🚀 Execution Pipeline

### 1. Dataset Generation
Prepare your training and evaluation sets using a specific `marking_strategy` (e.g., `weak`, `io`, `BA`, `BAS`).

**Create Training Set:**
Creates training data with respect to marking strategy.
```bash
python run_attrib.py createtraindataset \
    --model_name_or_path="/path/to/your_llama_model" \
    --path_dataset="/path/to/initial_training_data" \
    --dir_path_new_dataset="/path/to/save_processed_training_set" \
    --marking_strategy="your_marking_method"
```

**Create Evaluation Set:**
Creates evaluation data with respect to marking strategy.
```bash
python run_attrib.py createevaldataset \
    --model_name_or_path="/path/to/your_llama_model" \
    --path_dataset="/path/to/evaluation_data.jsonl" \
    --dir_path_new_dataset="/path/to/save_processed_eval_set.json" \
    --marking_strategy="your_marking_method"
```

### 2. Model Training
Perform Full Fine-Tuning.

```bash
python run_attrib.py train \
    --model_name_or_path="/path/to/your_llama_model" \
    --output_dir="/path/to/save_trained_model" \
    --logging_dir="/path/to/save_logs" \
    --dir_path_dataset="/path/to/processed_training_set" \
    --attn_implementation="flash_attention_2" \
    --marking_strategy="your_marking_method" \
    --alpha=0.5 \
    --training_batch_size=16 \
    --gradient_acc=2
```

### 3. Inference & Contributions
Compute the contribution of each document to the generated response. You can point to the general model directory or a specific checkpoint.

```bash
python run_attrib.py inf \
    --model_name_or_path="/path/to/your_llama_model" \
    --path_trained_model="/path/to/save_trained_model" \
    --path_savecontribs="/path/to/save_computed_contributions" \
    --marking_strategy="your_marking_method" \
    --path_dataset="/path/to/processed_eval_set" \
    --attn_implementation="flash_attention_2" \
    --alpha=0.5 \
    --checkpoint="checkpoint-XXX"
```

### 4. Attribution Aggregation
Convert token-level logit or probability contributions into final document citations.

**Multiple Aggregation (Grid Search):**
Performs multiple aggregations by testing multiple thresholds or methods simultaneously by providing comma-separated values.
```bash
python run_attrib.py multipleaggr \
    --model_name_or_path="/path/to/your_llama_model" \
    --path_savecontribs="/path/to/all_contributions" \
    --marking_strategy="vanilla" \
    --aggreg_method=max,avg,prop \
    --path_output="/path/to/all_attributions_aggregations_output" \
    --threshold_attr=4,5,6,7 \
    --threshold_count=0.5,0.75 \
    --contribs_to_use="logits" \
    --invfailsafe
```

### 5. Format for Trust-Align Evaluation
Finalize the attribution data to be compatible with standard Trust-Align master files (ASQA, ELI5, QAMPARI).

```bash
python run_attrib.py turntotrust \
    --asqa_file="/path/to/trustdata/asqa.json" \
    --eli5_file="/path/to/trustdata/eli5.json" \
    --qampari_file="/path/to/trustdata/qampari.json" \
    --folder_to_process="/path/to/save_final_attributions" \
    --folder_to_save="/path/to/trust_align_formatted_results"
```

---

## 📝 Key Arguments Reference

| Argument | Description |
| :--- | :--- |
| `--marking_strategy` | Strategy for tagging documents: `AW`, `BA`, `BAS`, `weak`. |
| `--alpha` | Scaling weight for the attribution loss component. |
| `--aggreg_method` | Strategy to pool contributions: `max` (peak score), `avg` (mean), or `prop` (proportion). |
| `--threshold_attr` | The numerical threshold above which a document is considered "cited". |
| `--contribs_to_use` | Choose between raw `logits` or normalized `probas` for scoring. Default `logits` |

---

## 📂 Project Structure
- `run_attrib.py`: Main CLI entry point.
- `attrib_training_fft.py`: Logic for Full Fine-Tuning.
- `attrib_inference.py`: Logic for extracting contributions during generation.
- `process_dataset.py`: Tools for marking documents and data formatting.
- `attrib_aggr.py`: Algorithms for aggregating token scores into citations.
- `turnALCEattribsintoTrustalign.py`: Data synchronization with master evaluation files.
- `utils.py` Helper functions for regex and text cleaning

---
### 🔗 Related Frameworks
If you are interested in the **Trust-Align** evaluation framework used in this pipeline, please visit the [official GitHub repository](https://github.com/declare-lab/trust-align).
---