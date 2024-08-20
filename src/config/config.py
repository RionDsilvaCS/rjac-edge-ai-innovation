import yaml
import os

CONFIG_PATH = "./src/config/experiments"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def build_config(exp_no: str):
    cfg = load_config(exp_no)
    return cfg