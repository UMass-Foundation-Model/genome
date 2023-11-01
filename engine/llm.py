from vllm import LLM, SamplingParams

class Wizardlm():
    @classmethod
    def init(cls, base_model="WizardLM/WizardCoder-Python-34B-V1.0", n_gpus=4, max_input_tokens=16384):
        cls.llm = LLM(model=base_model, tensor_parallel_size=n_gpus, max_num_batched_tokens=max_input_tokens)

    @classmethod
    def generate(cls, prompt, stop_token=None, temperature=0, top_p=1, max_new_tokens=2048):
        problem_instruction = [prompt]
        stop_tokens = ['</s>']
        if stop_token:
            stop_tokens.append(stop_token)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = cls.llm.generate(problem_instruction, sampling_params)
        return completions[0].outputs[0].text


class Codellama():
    @classmethod
    def init(cls, base_model="codellama/CodeLlama-34b-Python-hf", n_gpus=4, max_input_tokens=8192):
        cls.llm = LLM(
            model=base_model,
            dtype="float16",
            trust_remote_code=True,
            tensor_parallel_size=n_gpus,
            tokenizer="hf-internal-testing/llama-tokenizer",
            max_num_batched_tokens=max_input_tokens)

    @classmethod
    def generate(cls, prompt, stop_token=None, temperature=0, top_p=1, max_new_tokens=2048):
        problem_instruction = [prompt]
        stop_tokens = ['</s>']
        if stop_token:
            stop_tokens.append(stop_token)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, stop=stop_tokens)
        completions = cls.llm.generate(problem_instruction, sampling_params)
        return completions[0].outputs[0].text