import os
import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Union

import mlflow
import git


mlruns_path = os.path.join(__file__, "..", "..", "mlruns")
mlruns_path = os.path.abspath(mlruns_path)
mlflow.set_tracking_uri(mlruns_path)


def _get_current_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.commit.hexsha


def _experiment_id(name: str) -> int:
    client = mlflow.tracking.client.MlflowClient()
    for e in client.list_experiments():
        if e.name == name:
            return e.experiment_id
    raise KeyError("No experiment with matching name")


class ExperimentException(Exception):
    pass


@contextmanager
def start_run(experiment: Union[str,int], source_name: str, **kwargs):
    if isinstance(experiment, str):
        experiment_id = _experiment_id(experiment)
    elif isinstance(experiment, int):
        experiment_id = experiment
    else:
        raise ExperimentException("Provide experiment id or name")
    commit_hash = _get_current_commit_hash()
    with mlflow.start_run(
            experiment_id=experiment_id, 
            source_version=commit_hash, 
            source_name=source_name, 
            **kwargs) as run:
        yield run


def log_args(prefix: str=''):
    def _outer(func):
        argspec = inspect.getfullargspec(func)

        @wraps(func)
        def _inner(*args, **kwargs):
            # convert all args to kwargs
            for argname, argvalue in zip(argspec.args, args):
                if argname in kwargs:
                    raise TypeError(f"{func.__name__}() got multiple values for argument '{argname}'")
                kwargs[argname] = argvalue
            # extract default values for unspecified arguments
            defaults = argspec.defaults or []
            default_argnames = argspec.args[-len(defaults):]
            for argname, defaultvalue in zip(default_argnames, defaults):
                if argname not in kwargs:
                    kwargs[argname] = defaultvalue
            # kwargs contain all arguments now and we can log them to mlflow
            ignore_arg = lambda argname: argname.startswith("_") or argname == "self"
            for argname, argvalue in kwargs.items():
                if not ignore_arg:
                    mlflow.log_param(prefix + argname, argvalue)
            return func(**kwargs)

        return _inner
    return _outer
