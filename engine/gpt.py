from engine.utils import ProgramGenerator


def get_generator(args):
    if args.stage == 1:
        return ProgramGenerator(args)
    elif args.stage == 1.5:
        return ProgramGenerator(args)
    elif args.stage == 2:
        return ProgramGenerator(args)
    elif args.stage == 3:
        return ProgramGenerator(args)
    pass


def get_response(args, prompt_to_gpt):
    generator = get_generator(args)
    response, _ = generator.generate(prompt_to_gpt)
    response = response.replace('\t', '    ')
    return response


def parse_response(args, response: str):
    if args.stage == 1:
        if '"""' not in response:
            return False, None, None
        module_dict = {}


        ''' copied code '''
        prog_lines = response.split("\n")
        is_prog = False
        in_comment = False
        module_head = []
        for line in prog_lines:
            if '():' in line:
                is_prog = True
            if is_prog:
                module_head.append(line)
            if '"""' in line:
                if not in_comment:
                    in_comment = True
                else:
                    break
        module_name = module_head[0].split('class')[1].split('(')[0].strip()
        module_head = "\n".join(module_head)
        if module_head.count("class") > 1:
            module_head = module_head.rsplit("class", 1)[0]
        program = [(line.strip() if '=' in line else '')
                    for line in module_head.split("\n")]
        program = "\n".join(program).strip()


        module_dict['module_name'] = module_name
        module_dict['module_head'] = module_head
        return True, module_dict, program
    elif args.stage == 2:
        if 'step_name' in response:
            return '\n    step_name' + response.split('step_name', 1)[1]
        else:
            return response
    elif args.stage == 1.5 or args.stage == 3:
        program = [(line.strip() if '=' in line else '')
                    for line in response.split("\n")]
        program = "\n".join(program).strip()
        return program
