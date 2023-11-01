import os
from PIL import Image
import openai
import numpy as np
import copy
import io, tokenize
import math
import time

from .step_interpreters import parse_step
from engine.llm import Wizardlm, Codellama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from param import parse_opt

args, opt = parse_opt()

class Program:
    def __init__(self,prog_str,init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n') # 每一行的代码


class ProgramInterpreter:
    def __init__(self, step_interpreters):
        self.step_interpreters = step_interpreters

    def add_step_interpreter(self, step_name, interpreter):
        self.step_interpreters[step_name] = interpreter

    def execute_step(self,prog_step,inspect):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        print(step_name)
        args = parse_result['args']
        print(args)
        for key in args.keys():
            arg_str = args[key]
            if arg_str[-1] in ("'", '"'):
                if arg_str[0] == 'f':
                    arg_str = eval(arg_str[1:])
                    print(arg_str)
                    args[key] = arg_str.format(**prog_step.state)
                else:
                    args[key] = eval(arg_str)
            else:
                try:
                    args[key] = prog_step.state[arg_str]
                except Exception as e:
                    args[key] = eval(arg_str)
        #print(args)
        execute_result = self.step_interpreters[step_name].execute(*args.values())
        output_var = parse_result['output_var']
        prog_step.state[output_var] = execute_result
        return execute_result

    def execute(self,prog,init_state,inspect=False):
        if isinstance(prog,str):
            prog = Program(prog,init_state)
        else:
            assert(isinstance(prog,Program))

        prog_steps = [Program(instruction,init_state=prog.state) \
            for instruction in prog.instructions] #

        html_str = '<hr>'
        for prog_step in prog_steps: #
            if inspect:
                step_output, step_html = self.execute_step(prog_step,inspect)
                html_str += step_html + '<hr>'
            else:
                step_output = self.execute_step(prog_step,inspect)
            import pdb; pdb.set_trace()

        if inspect:
            return step_output, prog.state, html_str

        return step_output, prog.state # step_output


class ProgramGenerator():
    def __init__(self,args=None, temperature=0.0, top_p=0.5,prob_agg='mean'):
        with open('api.key') as f:
            openai.api_key = f.read().strip()
        self.temperature = args.temperature
        self.top_p = top_p
        self.prob_agg = prob_agg
        self.args = args
        self.model = args.model
        self.stop_token = args.stop_token

    def compute_prob(self,response):
        eos = '<|endoftext|>'
        for i,token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token==eos:
                break

        if self.prob_agg=='mean':
            agg_fn = np.mean
        elif self.prob_agg=='sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    @retry(wait=wait_random_exponential(min=0.2, max=0.5), stop=stop_after_attempt(10))
    def generate(self,inputs):
        if args.model == 'wizardlm':
            return Wizardlm.generate(inputs,self.stop_token), None
        elif args.model == 'codellama':
            return Codellama.generate(inputs,self.stop_token), None
        else:
            if args.model == 'gpt-3.5-turbo-16k':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": inputs + '\n'}], # ! ?? '\n' is Pending
                    temperature=self.temperature,
                    max_tokens=700)

                reply = response['choices'][0]['message']['content']
                total_tokens = response['usage']['total_tokens']
                # Pending
                prob = total_tokens # deprecated
                prog = reply.lstrip('\n').rstrip('\n')

                # remove potential space line
                prog2 = []
                for line in prog.split('\n'):
                    if line != '':
                        prog2.append(line)
                prog = '\n'.join(prog2)
            elif args.model == 'text-davinci-003' or args.model == 'gpt-3.5-turbo-instruct':
                response = openai.Completion.create(
                    model=args.model,
                    prompt=inputs,
                    # temperature=self.temperature,
                    temperature=0,
                    max_tokens=512,
                    # top_p=self.top_p,
                    top_p=0.5,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=1,
                    logprobs=1
                )

                prob = self.compute_prob(response)
                prog = response.choices[0]['text'].lstrip('\n').rstrip('\n')

            return prog, prob
