import argparse

# Importing subtask functions
from attrib_training_fft import start_training_fft
from attrib_inference import perform_multiple_inf
from process_dataset import create_train_dataset, create_eval_dataset
from attrib_aggr import perform_multiple_aggreg
from turnALCEattribsintoTrustalign import process_data

def main():
    parser = argparse.ArgumentParser(description="A simple function caller based on arguments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Training Subparser (Full Fine-Tuning) ---
    parser_train_fft = subparsers.add_parser("trainfft", help="Perform training")
    parser_train_fft.add_argument("--model_name_or_path", type=str, required=True, help="Path to original model")
    parser_train_fft.add_argument("--output_dir", type=str, required=True, help="Path to save model/checkpoints")
    parser_train_fft.add_argument("--logging_dir", type=str, required=True, help="Path to save logs")
    parser_train_fft.add_argument("--nepoch", type=int, default=1, help="Number of epochs")
    parser_train_fft.add_argument("--eos_token", type=str, default="<|eot_id|>")
    parser_train_fft.add_argument("--assistant_end_token", type=str, default="<|end_header_id|>")
    parser_train_fft.add_argument("--dir_path_dataset", type=str, required=True, help="Training dataset directory")
    parser_train_fft.add_argument("--path_eval_dataset", type=str, default="", help="Eval dataset path")
    parser_train_fft.add_argument("--training_batch_size", type=int, default=1)
    parser_train_fft.add_argument("--gradient_acc", type=int, default=16)
    parser_train_fft.add_argument("--attn_implementation", type=str, default="")
    parser_train_fft.add_argument("--marking_strategy", type=str, required=True)
    parser_train_fft.add_argument("--alpha", type=float, required=True, help="Balancing parameter for losses")

    # --- Training Dataset Creation Subparser ---
    parser_train_ds = subparsers.add_parser("createtraindataset", help="Create training dataset")
    parser_train_ds.add_argument("--model_name_or_path", type=str, required=True)
    parser_train_ds.add_argument("--path_dataset", type=str, required=True, help="Initial training datasets folder")
    parser_train_ds.add_argument("--dir_path_new_dataset", type=str, required=True, help="Where to save new dataset")
    parser_train_ds.add_argument("--nb_docs_per_query", type=int, default=5)
    parser_train_ds.add_argument("--marking_strategy", type=str, required=True)

    # --- Evaluation Dataset Creation Subparser ---
    parser_eval_ds = subparsers.add_parser("createevaldataset", help="Create evaluation dataset")
    parser_eval_ds.add_argument("--model_name_or_path", type=str, required=True)
    parser_eval_ds.add_argument("--path_dataset", type=str, required=True, help="Initial evaluation dataset folder")
    parser_eval_ds.add_argument("--dir_path_new_dataset", type=str, required=True)
    parser_eval_ds.add_argument("--nb_docs_per_query", type=int, default=5)
    parser_eval_ds.add_argument("--marking_strategy", type=str, required=True)

    # --- Contribution Aggregation Subparser ---
    parser_aggr = subparsers.add_parser("multipleaggr", help="Aggregate contributions into attributions")
    parser_aggr.add_argument("--model_name_or_path", type=str, required=True)
    parser_aggr.add_argument("--path_savecontribs", type=str, required=True, help="Folder containing contributions")
    parser_aggr.add_argument("--contrib_to_attribute", type=str, required=True, help="Filename of contributions")
    parser_aggr.add_argument("--aggreg_method", type=str, default="prop")
    parser_aggr.add_argument("--path_output", type=str, required=True, help="Where to save attributions")
    parser_aggr.add_argument("--threshold_attr", type=str, required=True)
    parser_aggr.add_argument("--threshold_count", type=str, required=True, help="Only useful for prop aggregation")
    parser_aggr.add_argument("--contribs_to_use", type=str, default="logits", help="probas or logits")
    parser_aggr.add_argument("--invfailsafe", action="store_true", help="Output specific refusal if no attribution")

    # --- Trust-Align Format Subparser ---
    parser_trust = subparsers.add_parser("turntotrust", help="Format data for trust-align evaluation")
    parser_trust.add_argument("--asqa_file", type=str, required=True)
    parser_trust.add_argument("--eli5_file", type=str, required=True)
    parser_trust.add_argument("--qampari_file", type=str, required=True)
    parser_trust.add_argument("--folder_to_process", type=str, required=True)
    parser_trust.add_argument("--folder_to_save", type=str, required=True)

    # --- Inference Subparser ---
    parser_inf = subparsers.add_parser("inf", help="Perform inference and compute contributions")
    parser_inf.add_argument("--model_name_or_path", type=str, required=True)
    parser_inf.add_argument("--path_trained_model", type=str, required=True)
    parser_inf.add_argument("--path_savecontribs", type=str, required=True)
    parser_inf.add_argument("--marking_strategy", type=str, required=True)
    parser_inf.add_argument("--path_dataset", type=str, required=True)
    parser_inf.add_argument("--inference_batch_size", type=int, default=1)
    parser_inf.add_argument("--nb_docs_per_query", type=int, default=5)
    parser_inf.add_argument("--eos_token", type=str, default="<|eot_id|>")
    parser_inf.add_argument("--assistant_end_token", type=str, default="<|end_header_id|>")
    parser_inf.add_argument("--attn_implementation", type=str, default="")
    parser_inf.add_argument("--alpha", type=str, required=True)
    parser_inf.add_argument("--checkpoint", type=str, default="")

    args = parser.parse_args()

    # Route execution to the correct function based on command
    if args.command == "trainfft":
        start_training_fft(
            args.model_name_or_path, args.output_dir, args.logging_dir, args.nepoch, 
            args.eos_token, args.assistant_end_token, args.dir_path_dataset, 
            args.path_eval_dataset, args.training_batch_size, args.gradient_acc, 
            args.attn_implementation, args.marking_strategy, args.alpha
        )

    elif args.command == "createtraindataset":
        create_train_dataset(
            args.model_name_or_path, args.path_dataset, args.dir_path_new_dataset, 
            args.nb_docs_per_query, args.marking_strategy
        )

    elif args.command == "createevaldataset":
        create_eval_dataset(
            args.model_name_or_path, args.path_dataset, args.dir_path_new_dataset, 
            args.nb_docs_per_query, args.marking_strategy
        )

    elif args.command == "inf":
        perform_multiple_inf(
            args.model_name_or_path, args.path_trained_model, args.path_dataset, 
            args.path_savecontribs, args.marking_strategy, args.eos_token, 
            args.attn_implementation, args.inference_batch_size, 
            args.alpha, args.checkpoint
        )

    elif args.command == "multipleaggr":
        perform_multiple_aggreg(
            args.model_name_or_path, args.path_savecontribs, args.contrib_to_attribute, 
            args.aggreg_method, args.path_output, args.threshold_attr, 
            args.threshold_count, args.contribs_to_use, args.invfailsafe
        )

    elif args.command == "turntotrust":
        process_data(
            args.asqa_file, args.eli5_file, args.qampari_file, 
            args.folder_to_process, args.folder_to_save
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()