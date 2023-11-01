"""
This is the script that contains the backend code. No need to look at this to implement new functionality
Functions that run separate processes. These processes run on GPUs, and are queried by processes running only CPUs
"""

import dill
import inspect
import queue
import torch
import torch.multiprocessing as mp
from rich.console import Console
from time import time
from typing import Callable, Union

from .configs import config


def make_fn(model_class, process_name, gpu_number):
    """
    model_class.name and process_name will be the same unless the same model is used in multiple processes, for
    different tasks
    """
    # We initialize each one on a separate GPU, to make sure there are no out of memory errors

    model_instance = model_class(gpu_number=gpu_number)

    def _function(*args, **kwargs):
        if process_name != model_class.name:
            kwargs['process_name'] = process_name

        if model_class.to_batch and not config.multiprocessing:
            # Batchify the input. Model expects a batch. And later un-batchify the output.
            args = [[arg] for arg in args]
            kwargs = {k: [v] for k, v in kwargs.items()}

            # The defaults that are not in args or kwargs, also need to listify
            full_arg_spec = inspect.getfullargspec(model_instance.forward)
            if full_arg_spec.defaults is None:
                default_dict = {}
            else:
                default_dict = dict(zip(full_arg_spec.args[-len(full_arg_spec.defaults):], full_arg_spec.defaults))
            non_given_args = full_arg_spec.args[1:][len(args):]
            non_given_args = set(non_given_args) - set(kwargs.keys())
            for arg_name in non_given_args:
                kwargs[arg_name] = [default_dict[arg_name]]

        try:
            out = model_instance.forward(*args, **kwargs)
            if model_class.to_batch and not config.multiprocessing:
                out = out[0]
        except Exception as e:
            print(f'Error in {process_name} model:', e)
            out = None
        return out

    return _function
