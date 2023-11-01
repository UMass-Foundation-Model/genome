from pprint import pprint
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings

    # Dataset and Image
    parser.add_argument('--dataset', type=str, default="gqa", help='') # Pending

    parser.add_argument('--ann_path', type=str, default="", help='')
    parser.add_argument('--image_path', type=str, default="", help='')
    parser.add_argument('--dataset_dir', type=str, default="", help='')
    parser.add_argument('--output_dir', type=str, default="", help='')
    parser.add_argument('--reuse_dir', type=str, default="", help='')
    parser.add_argument('--split', type=str, default="test", help='')

    parser.add_argument('--last_stage_output_dir', type=str, default="", help='')
    parser.add_argument('--threshold', type=float, default=0.5, help='')

    parser.add_argument('--coco_dir', type=str, default="", help='')

    parser.add_argument('--temperature', type=float, default=0, help='')
    parser.add_argument('--begin', type=int, default=0, help='')

    # Bool
    parser.add_argument('--use_new_module', action='store_true', default=False)
    parser.add_argument('--save_output', action='store_true', default=False)
    parser.add_argument('--add_cases', action='store_true', default=False)
    parser.add_argument('--split_cases', action='store_true', default=False)
    parser.add_argument('--save_all_module', action='store_true', default=False)
    parser.add_argument('--save_case_result', action='store_true', default=False)
    parser.add_argument('--save_prog_state', action='store_true', default=False)

    # Prompt
    parser.add_argument('--learning_prompt_path', type=str, help="", default='./prompts/learning_prompt_debug.txt')
    parser.add_argument('--module_make_prompt_path', type=str, help="", default='./prompts/module_make_prompt.txt')
    parser.add_argument('--online_prompt_path', type=str, help="", default='./prompts/learning_prompt_online.txt')
    parser.add_argument('--offline_prompt_path', type=str, help="", default='./prompts/learning_prompt_offlinev2.txt')
    parser.add_argument('--inference_prompt_path', type=str, help="", default='./prompts/learning_prompt_inference.txt')
    parser.add_argument('--training_prompt_path', type=str, help="", default='./prompts/module_debug_train_prompt.txt')

    parser.add_argument('--module_debug_init_prompt_path', type=str, help="", default='./prompts/module_debug_init_prompt.txt')
    parser.add_argument('--module_debug_execute_error_prompt_path', type=str, help="", default='./prompts/module_debug_execute_error_prompt.txt')
    parser.add_argument('--module_debug_execute_wrong_prompt_path', type=str, help="", default='./prompts/module_debug_execute_wrong_prompt.txt')
    parser.add_argument('--merge_prompt_path', type=str, help="", default='./prompts/merge_prompt.txt')

    # Save
    parser.add_argument('--module_save_dir', type=str, help="", default='output/gqa_train_eval1') # Pending need to specify
    # Debug
    parser.add_argument('--test_num', type=int, help="", default=3) # test 100 samples or 105

    # Model and Key Hyperparameter
    parser.add_argument('--stop_token', type=str, default="", help='')
    parser.add_argument('--model', type=str, help="GPT Model", default='gpt-3.5-turbo-16k') # Pending "gpt-3.5-turbo-16k-0613" or text-davinci-003
    parser.add_argument('--stage', type=float, help="", default=0) # Pending

    # parse
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args, opt

if __name__ == '__main__':

    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])



