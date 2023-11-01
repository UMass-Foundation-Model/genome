import os
import json
import openai
from engine.llm import Wizardlm
from engine.llm import Codellama
from engine.datasets import get_dataset

def strip_dict(dict):
    for k, v in dict.items():
        if isinstance(v, str):
            dict[k] = v.strip()
    return dict


def get_module_list(args):
    if not args.use_new_module:
        return []
    module_save_dir = args.module_save_dir
    if os.path.isdir(module_save_dir):
        file_list = os.listdir(module_save_dir)
        module_name_dict = {}
        for filename in file_list:
            # relieve the name constraint
            if 'MODULE' in filename and 'json' in filename or 'json' in filename:
                file_path = os.path.join(module_save_dir, filename)
                try:
                    module_dict = json.load(open(file_path))
                    module_dict = strip_dict(module_dict)
                    name_key = "module_name" if "module_name" in module_dict else "name"
                    module_name = module_dict[name_key]
                    module_dict['module_name'] = module_name
                    if 'module_program' not in module_dict:
                        module_dict['module_program'] = module_dict['module']
                    #if 'annotations' not in module_dict:
                    #    module_dict['annotations'] = module_dict['program']
                    if module_name not in module_name_dict or \
                            module_dict['test_accuracy'] > module_name_dict[module_name]['test_accuracy']:
                        module_name_dict[module_name] = module_dict
                except:
                    import pdb
                    pdb.set_trace()
                    
        module_list = []
        for module_dict in module_name_dict.values():
            if 'test_accuracy' not in module_dict:
                module_list.append(module_dict)
            elif  module_dict['test_accuracy'] >= args.threshold:
                module_list.append(module_dict)
    else:
        print("There is no available module directory: %s"%(module_save_dir))
        module_list = []
    return module_list


def save_output(args, output_dict, filename=None):
    output_dir = args.output_dir
    if args.stage == 1:
        output_path = os.path.join(output_dir, output_dict['annotations'][0]['id'] + '.json')
        json.dump(output_dict, open(output_path, 'w'), indent=2)
    elif args.stage == 1.5:
        if args.split_cases:
            module_head_list = output_dict.pop('module_head_list')
            for index, module_head in enumerate(module_head_list):
                output_dict['module_head'] = module_head
                output_path = os.path.join(output_dir, output_dict['annotations'][index]['id'] + '.json')
                json.dump(output_dict, open(output_path, 'w'), indent=2)
        else:
            output_path = os.path.join(output_dir, output_dict['annotations'][0]['id'] + '.json')
            json.dump(output_dict, open(output_path, 'w'), indent=2)
    elif args.stage == 2:
        if filename is None:
            filename = 'MODULE_' + output_dict['module_name'] + '.json'
        output_path = os.path.join(output_dir, filename)
        json.dump(output_dict, open(output_path, 'w'), indent=2)
    elif args.stage == 3:
        output_path = os.path.join(output_dir, 'result_' + output_dict['id'] + '.json')
        json.dump(output_dict, open(output_path, 'w'), indent=2)
    pass


def init_gpt(args):
    if args.model == 'wizardlm':
        Wizardlm.init()
    elif args.model == 'codellama':
        Codellama.init()
    else:
        with open('api.key') as f:
            openai.api_key = f.read().strip()


def pre_process(args):
    if args.save_output:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
    init_gpt(args)


def post_process(args):
    if args.save_output:
        if args.stage == 3:
            dataset_class = get_dataset(args.dataset)
            dataset_class.post_process(args)
