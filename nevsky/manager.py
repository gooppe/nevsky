import urllib.request
import hashlib
import tarfile
import json
import os
import logging

logger = logging.getLogger(__name__)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_model_dump(link: str, checksum: str, dump_foler: str):
    logger.info(f"Start downloading {link} dump file")

    tmp_file, _ = urllib.request.urlretrieve(link)
    if md5(tmp_file) != checksum:
        raise RuntimeError(f"Incorrect checksum for f{link}")
    else:
        logger.info(f"Checksum {tmp_file} is valid")

    with tarfile.open(tmp_file) as tar:
        tar.extractall(path=dump_foler)
        os.remove(tmp_file)
        logger.info("Model dump has been extracted")


def install_model(model_name: str, install_dir="dumps", config="config.json"):
    with open(config) as config_file:
        config = json.load(config_file)

    if model_name not in config["model_dumps"]:
        raise RuntimeError(f"Model dump {model_name} doesn't described")

    checksum = config["model_dumps"][model_name]["checksum"]

    model_dump_checksum = os.path.join(install_dir, model_name)
    if os.path.exists(model_dump_checksum):
        print(f"Dump {model_name} already exist")
        return

    os.makedirs(install_dir, exist_ok=True)
    link = config["model_dumps"][model_name]["link"]
    download_model_dump(link, checksum, install_dir)
