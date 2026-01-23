import os
from huggingface_hub import hf_hub_download

IS_LOCAL_FILES_ONLY = True

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    os.makedirs("../checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="../checkpoints", local_files_only=IS_LOCAL_FILES_ONLY)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="../checkpoints", local_files_only=IS_LOCAL_FILES_ONLY)

    return model_path, config_path