import os
import json
from scipy.stats import ttest_rel, wilcoxon

def get_qid(filename):
    """ Extracts the unique identifier from the filename based on naming conventions. """
    if "ASQA" in filename or "ELI5" in filename:
        return filename.split('_')[-1]
    else:
        # Maintaining original logic for other datasets (e.g. QAMPARI)
        return filename.split('json')[-1]

def perform_test(path_1, path_2, path_result):
    final_result = {}
    
    # Mapping datasets to their specific calibration metric keys
    calib_keys = {
        "ASQA": "calib_str_em_f1",
        "ELI5": "calib_claims_nli_f1",
        "QAMPARI": "calib_qampari_em_f1"
    }

    # Pre-index path_2 files by their QID for O(1) lookup
    path_2_files = {get_qid(f): f for f in os.listdir(path_2)}

    for dataset in ["ASQA", "ELI5", "QAMPARI"]:
        # Initialize lists for metrics
        metrics = {
            "ts1": [], "ts2": [],   # trust_scores
            "gc1": [], "gc2": [],   # answered_citation_f1
            "gr1": [], "gr2": [],   # macro_f1
            "ac1": [], "ac2": []    # calibration metrics
        }

        # Iterate through files in the first path
        for file_1 in os.listdir(path_1):
            if dataset not in file_1:
                continue
                
            qid = get_qid(file_1)
            
            # Match files based on QID
            if qid in path_2_files:
                file_2 = path_2_files[qid]
                
                with open(os.path.join(path_1, file_1), "r") as f1, \
                     open(os.path.join(path_2, file_2), "r") as f2:
                    data_1 = json.load(f1)
                    data_2 = json.load(f2)

                # Collect standard metrics
                metrics["ts1"].append(data_1["trust_score"])
                metrics["ts2"].append(data_2["trust_score"])
                metrics["gc1"].append(data_1["answered_citation_f1"])
                metrics["gc2"].append(data_2["answered_citation_f1"])
                metrics["gr1"].append(data_1["macro_f1"])
                metrics["gr2"].append(data_2["macro_f1"])

                # Collect dataset-specific calibration metric
                ac_key = calib_keys[dataset]
                metrics["ac1"].append(data_1[ac_key])
                metrics["ac2"].append(data_2[ac_key])

        # Perform statistical tests
        t_trust, p_trust = ttest_rel(metrics["ts1"], metrics["ts2"])
        t_gc, p_gc = ttest_rel(metrics["gc1"], metrics["gc2"])
        t_gr, p_gr = ttest_rel(metrics["gr1"], metrics["gr2"])
        t_ac, p_ac = ttest_rel(metrics["ac1"], metrics["ac2"])

        w_trust, wp_trust = wilcoxon(metrics["ts1"], metrics["ts2"])
        w_gc, wp_gc = wilcoxon(metrics["gc1"], metrics["gc2"])
        w_gr, wp_gr = wilcoxon(metrics["gr1"], metrics["gr2"])
        w_ac, wp_ac = wilcoxon(metrics["ac1"], metrics["ac2"])

        # Construct result dictionary for this dataset
        final_result[dataset] = {
            "t_stat_trust": t_trust, "p_value_trust": p_trust,
            "t_stat_gc": t_gc, "p_value_gc": p_gc,
            "t_stat_gr": t_gr, "p_value_gr": p_gr,
            "t_stat_ac": t_ac, "p_value_ac": p_ac,
            "w_stat_trust": w_trust, "wp_value_trust": wp_trust,
            "wt_stat_gc": w_gc, "wp_value_gc": wp_gc,
            "wt_stat_gr": w_gr, "wp_value_gr": wp_gr,
            "wt_stat_ac": w_ac, "wp_value_ac": wp_ac
        }

    # Save results
    os.makedirs(path_result, exist_ok=True)
    with open(os.path.join(path_result, 'statisticalres.json'), "w") as fw:
        json.dump(final_result, fw, indent=4)

# Example execution
if __name__ == "__main__":
    path_1 = "my_path_run_1"
    path_2 = "my_path_run_2"
    path_res = "my_path_res"
    perform_test(path_1, path_2, path_res)