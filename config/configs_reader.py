from omegaconf import OmegaConf
from os.path import join

from utils import utils
from config import const

configs_files_path = join(const.ROOT_DIR_PATH, 'config\\resources')

all_configs_files_names = utils.get_all_files_from_dir(configs_files_path)

print(f'Found config files: {all_configs_files_names}')

_global_config = OmegaConf.create()

for config_file in all_configs_files_names:
    file_conf = OmegaConf.load(config_file)
    _global_config = OmegaConf.merge(_global_config, file_conf)


def _get_config_files_by_env_id(env_id: str):
    env_configs_files_path = join(configs_files_path, "envs")
    file_name = join(env_configs_files_path, f"{env_id}.yaml")
    return OmegaConf.load(file_name)


def _get_config_files_by_device(device_id: str):
    files_path = join(configs_files_path, "devices")
    file_name = join(files_path, f"{device_id}.yaml")
    return OmegaConf.load(file_name)


def get_config(env_id: str, device='cpu'):
    config = OmegaConf.create()
    config = OmegaConf.merge(config, _global_config)
    config = OmegaConf.merge(config, _get_config_files_by_env_id(env_id))
    config = OmegaConf.merge(config, _get_config_files_by_device(device))
    print(OmegaConf.to_yaml(config))
    return config
