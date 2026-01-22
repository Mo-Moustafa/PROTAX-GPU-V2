from protax.classify import classify_file
from scripts.calibration import evaluate
import sys

if __name__ == "__main__":
    protax_args = sys.argv
    if len(protax_args) < 4:
        print(
            "Usage: python scripts/process_seqs.py [PATH_TO_QUERY_SEQUENCES] [PATH_TO_PARAMETERS] [PATH_TO_TAXONOMY_FILE]"
        )
    query_dir, model_dir, tax_dir = protax_args[1:4]    
    classify_file(query_dir, model_dir, tax_dir)
    evaluate("pyprotax_results.csv", "models/ref_db/list_of_labels_nodeIDs.txt", "")

    # testing
    # query_dir = r"/home/roy/Documents/PROTAX-dsets/30k_small/refs.aln"
    # classify_file(query_dir, "models/params/model.npz", "models/ref_db/taxonomy37k.npz")
