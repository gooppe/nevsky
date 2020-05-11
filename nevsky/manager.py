import urllib.request
import hashlib
import json
import logging
import os
import tarfile
import urllib.request

logger = logging.getLogger(__name__)


def md5(fname: str) -> str:
    """Calculate md5 hash of file.
    Args:
        fname (str): name of file.
    Returns:
        str: hash.
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_model_dump(link: str, checksum: str, dump_foler: str):
    """Download pretrained translation model dump.
    Args:
        link (str): dump link.
        checksum (str): dump's checksum.
        dump_foler (str): dump's directory to store.
    Raises:
        RuntimeError: raises if checksum is invalid.
    """
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
    """Install downloaded model.
    Args:
        model_name (str): model name.
        install_dir (str, optional): model installation directory. Defaults to "dumps".
        config (str, optional): model configuration file. Defaults to "config.json".
    Raises:
        RuntimeError: raises if checksum of downloaded model is invalid.
    """
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
