from typing import Any, Union

import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
import pickle

# unsafe lets you save numpy array too
yaml = YAML(typ='unsafe')

def read_yaml(fname: Union[str, Path]) -> Any:
    """Read the given file using YAML.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    with open(fname, 'r') as f:
        content = yaml.load(f)

    return content

def write_yaml(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using YAML format.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by YAML.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'w') as f:
        yaml.dump(obj, f)

def read_pickle(fname: Union[str, Path]) -> Any:
    """Read the given file using Pickle.

    Parameters
    ----------
    fname : str
        the file name.

    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    with open(fname, 'rb') as f:
        content = pickle.load(f)

    return content

def write_pickle(fname: Union[str, Path], obj: object, mkdir: bool = True) -> None:
    """Writes the given object to a file using pickle format.

    Parameters
    ----------
    fname : Union[str, Path]
        the file name.
    obj : object
        the object to write.
    mkdir : bool
        If True, will create parent directories if they don't exist.

    Returns
    -------
    content : Any
        the object returned by pickle.
    """
    if isinstance(fname, str):
        fpath = Path(fname)
    else:
        fpath = fname

    if mkdir:
        fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def train_test_split(x, y, rng, split=0.8):

    rand_inds = rng.permutation(np.arange(len(x)))
    split_idx = int(len(x) * split)
    train_inds = rand_inds[:split_idx]
    test_inds = rand_inds[split_idx:]

    return x[train_inds], y[train_inds], x[test_inds], y[test_inds]
