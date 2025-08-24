import json
import os
import tarfile
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import requests


PRETRAIN_DATA_URL = (
    "https://huggingface.co/datasets/iiTzEddy/LLaVA-Pretrain-595K/resolve/main"
)
INSTRUCT_DATA_URL = (
    "https://huggingface.co/datasets/iiTzEddy/LLaVA-Instruct-150K/resolve/main"
)


def setup_cache(base_url, jsonl_filename, images_filename, cache_dir="./cache"):
    """Download and extract LLaVA dataset files if needed."""
    os.makedirs(cache_dir, exist_ok=True)

    jsonl_path = os.path.join(cache_dir, jsonl_filename)
    images_dir = os.path.join(cache_dir, images_filename.replace(".tar", "_images"))

    _download_file_if_needed(f"{base_url}/{jsonl_filename}", jsonl_path, "JSONL")
    _download_and_extract_images_if_needed(
        f"{base_url}/{images_filename}", cache_dir, images_dir
    )

    return jsonl_path, images_dir


def _download_file_if_needed(url, filepath, desc):
    if os.path.exists(filepath):
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(filepath, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=f"Downloading {desc}"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _download_and_extract_images_if_needed(tar_url, cache_dir, images_dir):
    if os.path.exists(images_dir):
        return

    tar_filename = tar_url.split("/")[-1]
    tar_path = os.path.join(cache_dir, tar_filename)
    _download_file_if_needed(tar_url, tar_path, "images tar")
    _extract_tar_with_progress(tar_path, images_dir)
    os.remove(tar_path)


def _extract_tar_with_progress(tar_path, images_dir):
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, images_dir)


class LLaVAPretrainDataset(Dataset):
    def __init__(self, cache_dir="./cache"):
        jsonl_path, images_dir = setup_cache(
            PRETRAIN_DATA_URL, "llava_pretrain_595k.jsonl", "images.tar", cache_dir
        )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.images_dir, item["image"])
        image = Image.open(image_path)
        return {"image": image, "messages": item["messages"], "id": item["id"]}


class LLaVAInstructDataset(Dataset):
    def __init__(self, cache_dir="./cache"):
        jsonl_path, images_dir = setup_cache(
            INSTRUCT_DATA_URL, "llava_instruct_150k.jsonl", "images.tar", cache_dir
        )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.images_dir = images_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.images_dir, item["image"])
        image = Image.open(image_path)
        return {"image": image, "messages": item["messages"], "id": item["id"]}


if __name__ == "__main__":
    # pretrain_dataset = LLaVAPretrainDataset()
    # print(pretrain_dataset[0])

    instruct_dataset = LLaVAInstructDataset()
    print(instruct_dataset[0])
