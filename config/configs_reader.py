from omegaconf import OmegaConf
from os import listdir
from os.path import isfile, join

configs_files_path = 'D:/2022/thesis/docs/src/daif-surgical-robot-control/config/resources'

all_configs_files_names = [join(configs_files_path, f) for f in listdir(configs_files_path) if
                           isfile(join(configs_files_path, f))]

print(f'Found config files: {all_configs_files_names}')

config = OmegaConf.create()

for config_file in all_configs_files_names:
    file_conf = OmegaConf.load(config_file)
    config = OmegaConf.merge(config, file_conf)

print(OmegaConf.to_yaml(config))
