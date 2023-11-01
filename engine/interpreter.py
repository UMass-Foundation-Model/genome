import json
import traceback
from engine.step_interpreters import *
from engine.datasets import get_dataset


def create_interpreter(args):
    dataset_class = get_dataset(args.dataset)
    return dataset_class.get_interpreter(args)


def create_module_instance(args, interpreter, module_dict):

    module_name = module_dict['module_name']
    module_prog = module_dict['module_program']

    ''' copied code '''
    try:
        exec(module_prog, globals())
        interpreter.add_step_interpreter(
            module_name.upper(), eval(module_name + '()', globals()))
        print("successfully intialize the module %s!" %
                module_name.upper())
        if "VERIFY" in module_name and "META_VERIFY" in interpreter.step_interpreters:
            if not hasattr(interpreter.step_interpreters['META_VERIFY'], 'sub_module_dict'):
                interpreter.step_interpreters['META_VERIFY'].sub_module_dict = {}
            interpreter.step_interpreters['META_VERIFY'].sub_module_dict[module_name] = interpreter.step_interpreters[module_name]
        if "COMPARE" in module_name and "META_COMPARE" in interpreter.step_interpreters:
            if not hasattr(interpreter.step_interpreters['META_COMPARE'], 'sub_module_dict'):
                interpreter.step_interpreters['META_COMPARE'].sub_module_dict = {}
            interpreter.step_interpreters['META_COMPARE'].sub_module_dict[module_name] = interpreter.step_interpreters[module_name]
        if "SORT_SPATIAL" in module_name and "SORT_SPATIAL_OBJ" in interpreter.step_interpreters:
            if not hasattr(interpreter.step_interpreters['SORT_SPATIAL_OBJ'], 'sub_module_dict'):
                interpreter.step_interpreters['SORT_SPATIAL_OBJ'].sub_module_dict = {}
            interpreter.step_interpreters['SORT_SPATIAL_OBJ'].sub_module_dict[module_name] = interpreter.step_interpreters[module_name]
        return interpreter, None
    except Exception as e:
        print("ERROR when creating instance!")
        traceback_message = traceback.format_exc()
        print(traceback_message)
        error_line = None
        for traceback_line in traceback_message.splitlines():
            if 'File "<string>"' in traceback_line:
                error_line = traceback_line.strip()
        line_message = ""
        if error_line:
            lineno = int(error_line.split("line ")[1].split(',')[0])
            module_prog_lines = module_prog.splitlines()
            class_line = module_prog_lines[lineno - 1].strip(
            ) if lineno <= len(module_prog_lines) else None
            if class_line:
                error_line = error_line.replace(
                    'File "<string>"', f'Class {module_name}')
                line_message = error_line + '\n  ' + class_line + '\n'
        error_message = ""
        for traceback_line in traceback_message.splitlines():
            if 'Error:' in traceback_line:
                error_message = traceback_line.strip() + '\n'
        global_debug_output_exe = "\nDebuging message \n"
        debug_message = line_message + error_message
        global_debug_output_exe += debug_message
        # module_prog = (first_line + '\n' +
        #                module_prog).replace('\t', '    ')
        global_debug_output_exe += module_prog

        return interpreter, global_debug_output_exe


def execute_program(args, interpreter, ann):
    dataset_class = get_dataset(args.dataset)
    init_state = dataset_class.create_init_state(args, ann)

    program = ann['high_level_program']
    ''' copied code '''
    try:
        pred_answer, prog_state = interpreter.execute(program, init_state, inspect=False)
        pred_answer = dataset_class.process_result(args, pred_answer, prog_state, ann)
        for k, v in prog_state.items():
            try:
                prog_state[k] = json.dumps(v, indent=2)
            except Exception as e:
                prog_state[k] = v.__class__.__name__
        return pred_answer, None, prog_state

    except Exception as e:
        print("ERROR when executing program!")
        # print(e)
        traceback_message = traceback.format_exc()
        print(traceback_message)
        error_line = None
        for traceback_line in traceback_message.splitlines():
            if 'File "<string>"' in traceback_line:
                error_line = traceback_line.strip()
        line_message = ""
        error_message = ""
        for traceback_line in traceback_message.splitlines():
            if 'Error:' in traceback_line:
                error_message = traceback_line.strip() + '\n'
        global_debug_output_exe = "\nDebuging message \n"
        debug_message = line_message + error_message
        global_debug_output_exe += debug_message
        global_debug_output_exe += program

        return None, global_debug_output_exe, None


def verify_answer(args, pred_answer, ann):
    try:
        dataset_class = get_dataset(args.dataset)
        return dataset_class.verify_answer(args, pred_answer, ann)
    except Exception as e:
        print(e)
        return False


def test_on_cases(args, interpreter, test_cases):
    accuracy_list = []
    pred_answer_list = []
    message_list = []
    prog_state_list = []
    for case in test_cases:
        pred_answer, message, prog_state = execute_program(args, interpreter, case)
        accuracy_list.append(verify_answer(args, pred_answer, case))
        pred_answer_list.append(pred_answer)
        message_list.append(message)
        prog_state_list.append(prog_state)
    return accuracy_list, pred_answer_list, message_list, prog_state_list
