import os
import shutil

from ..configs.config import load_config
from ..dataset.mot import get_seq_names
from ..utils.log import log_or_print, get_logger


def benchmark(dataset, result_name, eval_root, result_root, seq_names, logger=None):
    """
        Copy results to TrackEval, evaluate then retrieve results
        Only train set is provided with gt
        Evaluates all sequences at once for now
    """
    # # Create seqmaps file
    # seqmap_path = os.path.join(eval_root, "data/gt/mot_challenge/seqmaps")
    # seqmap = f"{result_name}.txt"
    # seqmap_file = os.path.join(seqmap_path, seqmap)
    # if not os.path.isfile(seqmap_file):
    #     with open(seqmap_file, "w") as f:
    #         f.write("name\n")
    #     for i, name in enumerate(seq_names):
    #         with open(seqmap_file, "a+") as f:
    #             if i == len(seq_names) - 1:
    #                 f.write(name)
    #             else:
    #                 f.write(name + "\n")
    #     log_or_print(logger, f"Created seqmaps file at {seqmap_file}")
    # Copy results to TrackEval
    destination_folder = os.path.join(eval_root, "data/trackers/mot_challenge", f"{dataset}-train", result_name)
    data_folder = os.path.join(destination_folder, "data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
        print(os.path.isdir(data_folder))
    for name in seq_names:
        result_file = os.path.join(result_root, dataset, result_name, f"{name}.txt")
        target_file = os.path.join(data_folder, f"{name}.txt")
        assert os.path.isfile(result_file), f"No result file found at '{result_file}'"
        shutil.copyfile(result_file, target_file)
        log_or_print(logger, f"Copied {name}.txt for evaluation", "debug")
    # Evaluate and copy back results
    os.system(f"python {eval_root}/scripts/run_mot_challenge.py --USE_PARALLEL True \
        --TRACKERS_TO_EVAL {result_name} \
        --BENCHMARK {dataset} --METRICS CLEAR Identity")
    result_detail = os.path.join(destination_folder, "pedestrian_detailed.csv")
    result_summary = os.path.join(destination_folder, "pedestrian_summary.txt")
    shutil.copyfile(result_detail, os.path.join(result_root, dataset, result_name, f"{result_name}_result_detailed.csv"))
    shutil.copyfile(result_summary, os.path.join(result_root, dataset, result_name, f"{result_name}_result_summary.txt"))
    log_or_print(logger, f"Retrieved evaluation results and saved in {os.path.join(result_root, result_name)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate tracking result on benchmark")
    parser.add_argument("--result-name", default="TADAM_MOT17_train", type=str, help="result folder name")
    parser.add_argument("--config", default="TADAM_MOT17", type=str, help="config file")
    # parser.add_argument("--public-detection", default="all", choices=["all", "DPM", "FRCNN", "SDP"],
    #     type=str, help="test on specified public detection, valid for MOT17 only. default is all")
    # parser.add_argument("--sequence", default="all", type=str, help="test on specified sequence. default is all")
    args = parser.parse_args()

    config, cfg_msg = load_config(args.config)
    logger = get_logger(name="global", save_file=False, console_verbose=False)
    log_or_print(logger, cfg_msg)

    dataset = config.NAMES.DATASET
    full_seq_names, _, _ = get_seq_names(dataset, "train", "all", "all")

    benchmark(dataset, args.result_name, config.PATHS.EVAL_ROOT,
        config.PATHS.RESULT_ROOT, full_seq_names, logger)
