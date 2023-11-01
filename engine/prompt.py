

def get_prompt(args, module_list=[], type='default'):
    if args.stage == 1:
        prompt_path = args.online_prompt_path
        prompt = open(prompt_path).read().strip()

        new_module_head_list = []
        new_module_list = []
        for module_dict in module_list:
            new_module_head_list.append(module_dict['module_head'])
            new_module_list.append(module_dict['module_name'])

        prompt = prompt.replace('__NEW_MODULE_HEAD_LIST__', '\n'.join(new_module_head_list))
        prompt = prompt.replace('__NEW_MODULE_LIST__', '\n'.join(new_module_list))

        return prompt
    elif args.stage == 1.5:
        prompt_path = args.merge_prompt_path
        prompt = open(prompt_path).read().strip()

        new_module_head_list = []
        new_module_list = []
        for module_dict in module_list:
            new_module_head_list.append(module_dict['module_head'])
            new_module_list.append(module_dict['module_name'])

        prompt = prompt.replace('__NEW_MODULE_HEAD_LIST__', '\n'.join(new_module_head_list))
        prompt = prompt.replace('__NEW_MODULE_LIST__', '\n'.join(new_module_list))

        return prompt
    elif args.stage == 2:
        if type == 'default':
            prompt_path = args.module_make_prompt_path
        elif type == 'debug_init':
            prompt_path = args.module_debug_init_prompt_path
        elif type == 'debug_execute_error':
            prompt_path = args.module_debug_execute_error_prompt_path
        elif type == 'debug_execute_wrong':
            prompt_path = args.module_debug_execute_wrong_prompt_path
        prompt = open(prompt_path).read().strip()
        return prompt
    elif args.stage == 3:
        prompt_path = args.inference_prompt_path
        prompt = open(prompt_path).read().strip()

        new_module_example_list = []
        new_module_list = []
        for module_dict in module_list:
            if "annotations" not in module_dict:
                print("No examplar cases\n")
                continue
            for ann in module_dict["annotations"]:
                new_module_example_list.append(
                    f'Question: {ann["question"]}\n' +
                    'Program:\n' +
                    ann['high_level_program'])
            new_module_list.append(module_dict['module_name'])

        prompt = prompt.replace('__NEW_MODULE_EXAMPLE_LIST__', '\n'.join(new_module_example_list))
        prompt = prompt.replace('__NEW_MODULE_LIST__', '\n'.join(new_module_list))

        return prompt


def format_prompt(args, prompt, module=None, ann=None, message=None, pred_answer=None, type='default'):
    if args.stage == 1 or args.stage == 3:
        prompt = prompt.replace('__INSERT_NEW_QUESTION__', ann['question'])
        return prompt
    elif args.stage == 1.5:
        prompt = prompt.replace('__INSERT_NEW_QUESTION__', ann['question'])
        prompt = prompt.replace('__MODULE_NAME__', module['module_name'])
        prompt = prompt.replace('__MODULE_HEAD_HALF__', module['module_head_half'])
        return prompt
    elif args.stage == 2:
        if type == 'default':
            prompt = prompt.replace('__MODULE_NAME__', module['module_name'])
            prompt = prompt.replace('__MODULE_HEAD__', module['module_head'])
            return prompt
        elif type == 'debug_init':
            prompt = prompt.replace('__MODULE_NAME__', module['module_name'])
            prompt = prompt.replace('__MODULE_HEAD__', module['module_head'])
            prompt = prompt.replace('__DEBUG_MESSAGE__', message)
            return prompt
        elif type == 'debug_execute_error':
            prompt = prompt.replace('__MODULE_NAME__', module['module_name'])
            prompt = prompt.replace('__MODULE_HEAD__', module['module_head'])
            prompt = prompt.replace('__DEBUG_MESSAGE__', message)
            prompt = prompt.replace('__HIGH_LEVEL_PROGRAM__', ann['high_level_program'])
            return prompt
        elif type == 'debug_execute_wrong':
            prompt = prompt.replace('__MODULE_NAME__', module['module_name'])
            prompt = prompt.replace('__MODULE_HEAD__', module['module_head'])
            prompt = prompt.replace('__ANSWER__', ann['answer'])
            if not isinstance(pred_answer, str):
                pred_answer = 'not a string'
            prompt = prompt.replace('__OUTPUT__', pred_answer)
            prompt = prompt.replace('__HIGH_LEVEL_PROGRAM__', ann['high_level_program'])
            return prompt
