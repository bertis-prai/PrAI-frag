import os
import shutil
import yaml
from easydict import EasyDict as edict

def load(config_path, verbose=False):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    if verbose:
        with open(config_path, 'r', encoding='utf-8') as f:
            print()
            print('------------------------------ yml ------------------------------')
            for line in f:
                line = line.replace('\n', '')
                try:
                    print(line)
                except:
                    print('korean words not allowed')
            print('------------------------------ yml ------------------------------')
            print()

    return config


def save_config(config_path, train_dir):
    configs = [config for config in os.listdir(train_dir)
               if config.startswith('configs') and config.endswith('.yml')]
    if not configs:
        last_num = -1
    else:
        last_config = list(sorted(configs))[-1]  # ex) config5.yml
        last_num = int(last_config.split('.')[0].split('configs')[-1])

    save_name = 'configs%02d.yml' % (int(last_num) + 1)

    shutil.copy(config_path, os.path.join(train_dir, save_name))
