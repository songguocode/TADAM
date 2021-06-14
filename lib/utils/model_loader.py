import os
import torch
from collections import OrderedDict
from ..utils.log import log_or_print


def load_model(model, model_path, logger=None):
    # Check file
    if not os.path.isfile(model_path):
        log_or_print(logger, f"Invalid model at '{model_path}'. Please check path of file", level="critical")
        exit(0)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        log_or_print(logger, "Model file does not contain 'state_dict' or 'model'. Check again.", level="critical")
        exit(0)

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        # Discard "module." characters which is caused by parallel training
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        log_or_print(
            logger, f"The pretrained weights '{model_path}' cannot be loaded "
            "as no matched layers are found, please check the key names carefully.",
            level="critical"
        )
        exit(0)
    else:
        log_or_print(logger, f"Successfully loaded pretrained weights from '{model_path}'")
        if len(discarded_layers) > 0:
            log_or_print(logger, "** The following layers are discarded "
                f"due to unmatched keys or layer size: '{discarded_layers}'", level="warning")
    return model
