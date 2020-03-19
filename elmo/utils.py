import collections
import codecs
import json
import os

def dict2namedtuple(dic):
    return collections.namedtuple('Namespace', dic.keys())(**dic)

def load_config(model_dir='./elmo/configs/'):
    args2 = dict2namedtuple(
        json.load(
            codecs.open(
                os.path.join(model_dir, 'config.json'),
                'r', encoding='utf-8')
        )
    )
    with open(os.path.join(model_dir, args2.config_path), 'r') as config_json:
        config = json.load(config_json)
    return config
