from huggingface_hub import snapshot_download
import os

# Configuration
FLOAT = True
BINARY = True

# Gather pre-existing embeddings

if FLOAT:

    # Setup transaction details
    # Float32
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

    # Subfolder in the repo of the dataset where the file is stored
    folder_in_repo = "data"
    allow_patterns = f"{folder_in_repo}/*.parquet"

    # Where to store the local copy of the dataset
    local_dir = repo_id

    # Set repo type
    repo_type = "dataset"

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # Download the repo
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)

if BINARY:

    # Binary (bit packed)
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

    # Subfolder in the repo of the dataset where the file is stored
    folder_in_repo = "data"
    allow_patterns = f"{folder_in_repo}/*.parquet"

    # Where to store the local copy of the dataset
    local_dir = repo_id

    # Set repo type
    repo_type = "dataset"

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # Download the repo
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)
