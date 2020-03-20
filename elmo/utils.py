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


def read_dictionary(model_dir='./elmo/dictionary/', is_char=True):
    if isinstance(is_char, bool):
        dic_name = 'char.dic' if is_char else 'word.dic'
    else:
        if (is_instance(is_char, str) and is_char.split('.')[-1] == 'dic'):
            dic_name = is_char
        else:
            raise AttributeError("")
    dic_file_path = os.path.join(model_dir, dic_name)
    fpi = codecs.open(dic_file_path, 'r', encoding='utf-8')
    return fpi
