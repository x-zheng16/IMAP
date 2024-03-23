import csv
import datetime
import os
import random
import re
from collections import defaultdict
from distutils.util import strtobool
from functools import wraps

import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch import nn
from torch.distributions import Categorical, Independent, Normal

auto_bool = lambda x: bool(strtobool(x) if isinstance(x, str) else x)  # noqa: E731
MLP_NORM = {"BN": nn.BatchNorm1d, "LN": nn.LayerNorm, "None": None}


def make_session(g=None):
    import tensorflow as tf

    sess = tf.get_default_session()
    if sess is None:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1
        )
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
        sess = tf.Session(config=tf_config)
        sess.__enter__()
    return sess


def set_seed(seed):
    # pytorch
    torch.manual_seed(seed)

    # according to https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.compat.v1.set_random_seed(seed)
    except:  # noqa: E722
        pass


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def time_spent(time):
    output = str(datetime.timedelta(seconds=int(time)))
    return output


def time_remain(time, epoch, nepoch, last_epoch=0):
    time = time / (epoch - last_epoch) * (nepoch - epoch)
    output = "remain " + str(datetime.timedelta(seconds=int(time)))
    return output


class LogIt:
    def __init__(self, logfile="out.log"):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with open(self.logfile, mode="a", encoding="utf-8") as opened_file:
                output = list(map(str, args)) if len(args) else []
                output += [f"{k}={v}" for k, v in kwargs.items()]
                opened_file.write(", ".join(output) + "\n")
            return func(*args, **kwargs)

        return wrapped_function


def grad_monitor(net):
    grad_norm = 0
    for name, param in net.named_parameters():
        if param.grad is not None:
            if torch.all(~torch.isnan(param.grad)):
                grad_norm += torch.sum(param.grad.detach() ** 2)
            else:
                print(f"grad of param {name} is nan")
    if not np.isscalar(grad_norm):
        grad_norm = grad_norm.item()
    return grad_norm


def param_norm_monitor(net):
    param_norm = 0
    for name, param in net.named_parameters():
        if torch.all(~torch.isnan(param)):
            param_norm += torch.sum(param.detach() ** 2)
        else:
            print(f"param {name} is nan")
    if not np.isscalar(param_norm):
        param_norm = param_norm.item()
    return param_norm


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def last_layer_init(net):
    for m in getattr(net, "last", getattr(net, "mu", None)).modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.copy_(0.01 * m.weight.data)


def sync_weight(target_net, online_net, tau):
    for o, n in zip(target_net.parameters(), online_net.parameters()):
        o.data.copy_(o.data * tau + n.data * (1 - tau))


def disable_grad(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_grad(model):
    for p in model.parameters():
        p.requires_grad = True


def get_env_space(env, index=None):
    ob_space = env.observation_space
    ac_space = env.action_space
    if isinstance(ob_space, list):
        ob_space = ob_space[0]
        ac_space = ac_space[0]
    if hasattr(ob_space, "spaces"):
        ob_space = ob_space.spaces[index or 0]
        ac_space = ac_space.spaces[index or 0]
    return ob_space, ac_space


def split_batch(minibatch_size, batch_size, shuffle=False):
    indices = np.random.permutation(batch_size) if shuffle else np.arange(batch_size)
    for idx in range(0, batch_size, minibatch_size):
        if idx + minibatch_size * 2 >= batch_size:
            yield indices[idx:]
            break
        yield indices[idx : idx + minibatch_size]


def get_dist_fn(ac_space):
    if isinstance(ac_space, Discrete):
        dist_fn = Categorical
    elif isinstance(ac_space, Box):
        dist_fn = lambda *logits: Independent(Normal(*logits), 1)  # noqa: E731
    return dist_fn


def find_all_files(
    root_dir,
    pattern,
    suffix=None,
    prefix=None,
    return_pattern=False,
    exclude_suffix=(".png", ".txt", ".log", "config.json", ".pdf", ".yml"),
):
    file_list = []
    pattern_list = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            if suffix and not f.endswith(suffix):
                continue
            else:
                if f.endswith(exclude_suffix):
                    continue
            if prefix and not f.startswith(prefix):
                continue
            absolute_path = os.path.join(dirname, f)
            m = re.search(pattern, absolute_path)
            if m is not None:
                file_list.append(absolute_path)
                pattern_list.append(m.groups())
    if return_pattern:
        return file_list, pattern_list
    else:
        return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        m = re.search(pattern, f) if pattern else None
        res[m.group(1) if m else ""].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}
