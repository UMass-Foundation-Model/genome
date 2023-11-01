import os
import json
from engine.util import strip_dict
from engine.datasets import get_dataset

def get_samples(args):
    samples = []
    if args.stage == 1 or args.stage == 3:
        dataset_class = get_dataset(args.dataset)
        return dataset_class.get_samples(args)
    elif args.stage == 1.5 or args.stage == 2:
        last_stage_output_dir = args.last_stage_output_dir
        file_list = os.listdir(last_stage_output_dir)
        for filename in file_list:
            if 'json' in filename:
                file_path = os.path.join(last_stage_output_dir, filename)
                last_stage_output_dict = json.load(open(file_path))
                last_stage_output_dict = strip_dict(last_stage_output_dict)
                samples.append(last_stage_output_dict)

    return samples
