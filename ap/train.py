import warnings

# depress annoying warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import imp  # noqa: F401
    from collections import Container  # noqa: F401

import json
import os
import pprint
from os.path import dirname, join

import hydra
import torch
from gym.spaces import Box
from tianshou.data import VectorReplayBuffer

from ap import WORKPLACE
from ap.collector import Collector
from ap.conf.config import hydra_decorator, register_configs
from ap.envs.make_venv import make_venv
from ap.net import get_model
from ap.policy import POLICY_DICT
from ap.trainer import onpolicy_trainer
from ap.util import *  # noqa: F403

pp = pprint.PrettyPrinter(indent=4)


@hydra.main(config_path=None, config_name="config")
@hydra_decorator
def train(cfg):
    # seed
    set_seed(cfg.seed)

    # env
    venv, test_venv = make_venv(cfg)

    # model
    ob_space, ac_space = get_env_space(venv)
    encoder, actor, critic = get_model(ob_space, ac_space, **vars(cfg.r), **vars(cfg.m))

    # init
    if not cfg.eval:
        if hasattr(actor, "sigma_param"):
            torch.nn.init.constant_(actor.sigma_param, cfg.init_logsigma)
        actor.apply(weight_init)
        critic.apply(weight_init)
        if isinstance(ac_space, Box) and cfg.last_a:
            last_layer_init(actor)
        last_layer_init(critic)

    optim = torch.optim.Adam(actor.parameters(), lr=cfg.lr) if not cfg.eval else None

    # policy
    kwargs_policy = dict(
        encoder=encoder,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=get_dist_fn(ac_space),
        observation_space=ob_space,
        action_space=ac_space,
        **vars(cfg.r),
        **vars(cfg.p),
        minibatch_size=cfg.c.minibatch_size,
        total_updates=cfg.c.total_updates,
    )
    policy = POLICY_DICT[cfg.method](**kwargs_policy)

    # collector
    train_c = Collector(policy, venv, VectorReplayBuffer(cfg.c.buffer_size, len(venv)))
    test_c = Collector(
        policy, test_venv, VectorReplayBuffer(cfg.c.buffer_size, len(test_venv))
    )

    # logger
    logger = set_logger(cfg)

    # trainer
    def save_best_fn(policy):
        # state = {"model": policy.state_dict(), "obs_rms": venv.get_obs_rms()}
        # torch.save(state, join(cfg.log_dir, "policy_best.pth"))
        pass

    def save_checkpoint_fn():
        state = {"model": policy.state_dict(), "obs_rms": venv.get_obs_rms()}
        torch.save(state, join(os.getcwd(), "policy_latest.pth"))

    if not cfg.eval:
        result = onpolicy_trainer(
            policy=policy,
            train_collector=train_c,
            test_collector=test_c,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            save_state_freq=cfg.save_state_freq,
            state_map=cfg.state_map,
            test_after_train=cfg.test_after_train,
            verbose=cfg.verbose,
            **vars(cfg.c),
        )
        simple_result = {}
        for k in [
            "duration",
            "test_time",
            "train_time/collector",
            "train_time/model",
            "train_speed",
        ]:
            simple_result.update({k: result[k]})
        pp.pprint(simple_result)

    test(cfg, policy, test_venv, test_c, logger)

    # close
    venv.close()
    test_venv.close()


def test(cfg, policy, test_venv, test_c, logger):
    # load
    if cfg.resume_path is not None:
        path_list = find_all_files(cfg.resume_path, "policy_latest.pth")
        path_list.sort()
        print(f"find {len(path_list)} .pth files")
    elif cfg.method not in ["zero", "random"] and cfg.tag is not None:
        print("override path with zoo[task_type/task][tag][path]")
        with open(f"{WORKPLACE}/zoo/{cfg.task_type}/agents.json", "r") as f:
            zoo = json.load(f)
        path_list = [zoo[f"{cfg.task_type}/{cfg.task}"][cfg.tag]["path"]]
    else:
        path_list = [None]  # reserve to eval directly after train

    result_list = {}
    for path in path_list:
        if path is not None:
            ckpt = torch.load(path, map_location=cfg.m.device)
            state_dict = policy.state_dict()
            state_dict.update(
                {k: v for k, v in ckpt["model"].items() if k in state_dict}
            )
            policy.load_state_dict(state_dict)
            test_venv.set_obs_rms(ckpt["obs_rms"])
            print("Load agent from: ", path)

        # eval
        policy.eval()
        test_c.reset()
        n_episode = cfg.c.episode_per_test
        result = test_c.collect(n_episode=n_episode)
        result.update(
            {
                "test_time": f"{test_c.collect_time:.2f}s",
                "test_speed": f"{test_c.collect_step / test_c.collect_time:.2f} step/s",
            }
        )
        result_list[path] = result

        # with open(join(logger.writer.log_dir, f"result_{n_episode}.json"), "a") as f:
        #     json.dump(result, f, indent=4)
        # pickle.dump(
        #     test_c.buffer, open(join(logger.writer.log_dir, "test_buffer.pkl"), "wb")
        # )

        pp.pprint(
            {
                k: v
                for k, v in result.items()
                if k in ["n/ep", "rew", "rew_std", "len", "len_std", "success_rate"]
            }
        )
        # save result to each model path
        if path is not None:
            with open(join(dirname(path), f"result_{n_episode}.json"), "w") as f:
                json.dump(result, f, indent=4)


if __name__ == "__main__":
    register_configs()
    train()
