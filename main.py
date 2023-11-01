import numpy as np
from tqdm import tqdm
from param import parse_opt
from engine.dataset import get_samples
from engine.util import get_module_list, save_output, pre_process, post_process
from engine.gpt import get_response, parse_response
from engine.prompt import get_prompt, format_prompt
from engine.interpreter import create_interpreter, create_module_instance, test_on_cases


def stage1(args):
    module_list = get_module_list(args)
    prompt = get_prompt(args, module_list=module_list)
    samples = get_samples(args)

    for index, sample in enumerate(tqdm(samples)):
        prompt_to_gpt = format_prompt(args, prompt, ann=sample)
        response = get_response(args, prompt_to_gpt)
        make_module_flag, module_dict, high_level_program = parse_response(args, response)
        sample['high_level_program'] = high_level_program

        if args.save_output and make_module_flag:
            output_dict = {**module_dict, "annotations": [sample]}
            save_output(args, output_dict)


def stage1_5(args):
    samples = get_samples(args)

    module_head_dict = {}
    if args.add_cases:
        module_name_dict = {}
    for index, sample in enumerate(tqdm(samples)):
        module_head_half = sample['module_head'].split('Examples:')[0]
        if module_head_half in module_head_dict:
            output_dict = module_head_dict[module_head_half]
            output_dict['annotations'].extend(sample['annotations'])
        else:
            output_dict = sample
        if args.split_cases:
            if 'module_head_list' in output_dict:
                output_dict['module_head_list'].append(sample['module_head'])
            else:
                output_dict['module_head_list'] = [sample['module_head']]
        module_head_dict[module_head_half] = output_dict
        if args.add_cases:
            module_name = sample['module_name']
            ann_list = module_name_dict[module_name] if module_name in module_name_dict else []
            ann_list.extend(sample['annotations'])
            module_name_dict[module_name] = ann_list

    if args.add_cases:
        module_list = get_module_list(args)
        prompt = get_prompt(args, module_list=module_list)

    for module_head_half, output_dict in tqdm(module_head_dict.items()):
        if args.add_cases:
            module_name = output_dict['module_name']
            ann_list = module_name_dict[module_name]
            max_case_added = args.max_case_added
            for ann in ann_list:
                if max_case_added > 0 and ann not in output_dict['annotations']:
                    max_case_added -= 1
                    module_dict = {
                        "module_name": module_name,
                        'module_head_half': module_head_half
                    }
                    prompt_to_gpt = format_prompt(args, prompt, module=module_dict, ann=ann)
                    response = get_response(args, prompt_to_gpt)
                    ann['high_level_program'] = parse_response(args, response)
                    output_dict['annotations'].append(ann)

        if args.save_output:
            save_output(args, output_dict)


def stage2(args):
    interpreter = create_interpreter(args)
    module_list = get_module_list(args)

    for module_dict in module_list:
        interpreter, message = create_module_instance(args, interpreter, module_dict)
        assert not message, message

    prompt = get_prompt(args)
    samples = get_samples(args)

    for index, sample in enumerate(tqdm(samples)):
        prompt_to_gpt = format_prompt(args, prompt, module=sample)
        response = get_response(args, prompt_to_gpt)
        module_body = parse_response(args, response)
        sample['module_program'] = sample['module_head'] + module_body

        interpreter, message = create_module_instance(args, interpreter, sample)
        if message is None:
            test_cases = sample['annotations']
            accuracy_list, pred_answer_list, message_list, prog_state_list = test_on_cases(args, interpreter, test_cases)
            accuracy = float(np.mean(accuracy_list))
            if args.save_case_result:
                case_result = []
                for pred_answer, message in zip(pred_answer_list, message_list):
                    case_result.append(message if pred_answer is None else pred_answer)
                sample['case_result'] = case_result
            if args.save_all_module:
                output_dict = {**sample, 'test_accuracy': accuracy}
                filename = f'MODULE_{output_dict["module_name"]}_{index}.json'
                save_output(args, output_dict, filename=filename)
            if accuracy >= args.threshold:
                if args.save_output:
                    output_dict = {**sample, 'test_accuracy': accuracy}
                    save_output(args, output_dict)


def stage3(args):
    interpreter = create_interpreter(args)
    module_list = get_module_list(args)

    for module_dict in module_list:
        interpreter, message = create_module_instance(args, interpreter, module_dict)
        assert not message, message

    prompt = get_prompt(args, module_list=module_list)
    samples = get_samples(args)

    acc = 0.0
    cnt = 0
    for index, sample in enumerate(tqdm(samples)):
        prompt_to_gpt = format_prompt(args, prompt, ann=sample)
        response = get_response(args, prompt_to_gpt)

        sample['high_level_program'] = parse_response(args, response)

        test_cases = [sample]
        accuracy_list, pred_answer_list, message_list, prog_state_list = test_on_cases(args, interpreter, test_cases)
        accuracy = float(np.mean(accuracy_list))
        acc += accuracy
        cnt += 1
        if args.save_output:
            if pred_answer_list[0]:
                output_dict = {**sample, 'pred_answer': pred_answer_list[0], 'message': '', 'accuracy': accuracy}
            else:
                output_dict = {**sample, 'pred_answer': '', 'message': message_list[0], 'accuracy': accuracy}
            if args.save_prog_state:
                if prog_state_list[0]:
                    output_dict['prog_state'] = prog_state_list[0]
                else:
                    output_dict['prog_state'] = ""
            save_output(args, output_dict)
    acc = acc / cnt * 100
    print(f"Overall Accuracy: {acc}%\n")


def main():
    args, _ = parse_opt()
    pre_process(args)

    if args.stage == 1:
        stage1(args)
    elif args.stage == 1.5:
        stage1_5(args)
    elif args.stage == 2:
        stage2(args)
    elif args.stage == 3:
        stage3(args)

    post_process(args)

if __name__ == "__main__":
    main()
