import time
from collections import defaultdict

import tianshou
import torch
from tianshou.trainer import test_episode
from tianshou.utils import LazyLogger

from ap.util.utils import LogIt, time_remain


class BaseTrainer(tianshou.trainer.BaseTrainer):
    def __init__(
        self,
        learning_type,
        policy,
        max_epoch,
        minibatch_size,
        train_collector=None,
        test_collector=None,
        buffer=None,
        step_per_epoch=None,
        repeat_per_collect=None,
        episode_per_test=None,
        update_per_step=1,
        update_per_epoch=None,
        step_per_collect=None,
        episode_per_collect=None,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_best_fn=None,
        save_checkpoint_fn=None,
        resume_from_log=False,
        reward_metric=None,
        logger=LazyLogger(),
        verbose=True,
        show_progress=False,
        test_in_train=True,
        save_fn=None,
        **kwargs,
    ):
        super().__init__(
            learning_type,
            policy,
            max_epoch,
            minibatch_size,
            train_collector,
            test_collector,
            buffer,
            step_per_epoch,
            repeat_per_collect,
            episode_per_test,
            update_per_step,
            update_per_epoch,
            step_per_collect,
            episode_per_collect,
            train_fn,
            test_fn,
            stop_fn,
            save_best_fn,
            save_checkpoint_fn,
            resume_from_log,
            reward_metric,
            logger,
            verbose,
            show_progress,
            test_in_train,
            save_fn,
        )
        self.minibatch_size = minibatch_size
        self.log_dir = logger.writer.log_dir
        self.env_name = train_collector.env.spec[0].id


class OnpolicyTrainer(BaseTrainer):
    def __init__(
        self, save_state_freq=False, state_map=False, test_after_train=False, **kwargs
    ):
        super().__init__("onpolicy", **kwargs)
        self.state_freq = defaultdict(list)
        self.last_state_freq = defaultdict(list)
        self.save_state_freq = save_state_freq
        self.state_map = state_map
        self.success_rate = self.test_success_rate = 0.0
        self.final_state = {}
        self.test_rew = 0.0
        self.test_after_train = test_after_train

    def __next__(self):
        if self.iter_num > 1:
            if self.epoch > self.max_epoch:
                if self.save_state_freq:
                    torch.save(self.state_freq, self.log_dir + "/state_freq.pth")
        result = super().__next__()

        @LogIt(self.log_dir + "/output.log")
        def save_info(*args, **kwargs):
            print(*args, **kwargs)

        rtime = time_remain(
            time.time() - self.start_time,
            self.epoch,
            self.max_epoch,
            self.start_epoch,
        )
        info = f"#{self.epoch:<2} train | rew: {self.last_rew:8.2f} | sr: {self.success_rate:4.0%} | len: {self.last_len:4.0f} | {rtime}"
        if self.verbose:
            info += f" | {self.final_state} | in_rew_coef: {self.in_rew_coef:.4f}"
        if self.test_after_train:
            info += f" ||| test | rew: {self.test_rew:8.2f} | sr: {self.test_success_rate:4.0%} | len: {self.test_len:4.0f}"
            if self.verbose:
                info += f" | {self.test_final_state}"
        save_info(info)
        return result

    def test_step(self):
        stop_fn_flag = False
        test_stat = {}

        if self.test_after_train:
            test_result = test_episode(
                self.policy,
                self.test_collector,
                self.test_fn,
                self.epoch,
                self.episode_per_test,
                self.logger,
                self.env_step,
                self.reward_metric,
            )
            rew, rew_std = test_result["rew"], test_result["rew_std"]
            if self.best_epoch < 0 or self.best_reward < rew:
                self.best_epoch = self.epoch
                self.best_reward = float(rew)
                self.best_reward_std = rew_std
                if self.save_best_fn:
                    self.save_best_fn(self.policy)
            if not self.is_run:
                test_stat = {
                    "test_reward": rew,
                    "test_reward_std": rew_std,
                    "best_reward": self.best_reward,
                    "best_reward_std": self.best_reward_std,
                    "best_epoch": self.best_epoch,
                }
            if self.stop_fn and self.stop_fn(self.best_reward):
                stop_fn_flag = True
            self.test_rew = test_result["rew"]
            self.test_len = test_result["len"]
            self.test_success_rate = test_result["success_rate"]
            self.test_final_state = test_result["final_state"]
        return test_stat, stop_fn_flag

    def train_step(self):
        data, result, stop_fn_flag = super().train_step()
        if result["n/ep"] > 0:
            self.success_rate = result["success_rate"]
            self.final_state = result["final_state"]
        return data, result, stop_fn_flag

    def policy_update_fn(self, data, result=None):
        assert self.train_collector is not None
        learn_info = self.policy.update(
            0,
            self.train_collector.buffer,
            minibatch_size=self.minibatch_size,
            repeat=self.repeat_per_collect,
        )
        self.logger.log_info(learn_info["batch"], self.env_step)
        state = learn_info.pop("batch").info["state"]
        for k, v in state.items():
            v = v.cpu().numpy()
            self.state_freq[k].append(v)
            self.last_state_freq[k] = [v]
        self.train_collector.reset_buffer(keep_statistics=True)
        step = max([1] + [len(v) for v in learn_info.values() if isinstance(v, list)])
        self.gradient_step += step
        self.in_rew_coef = learn_info["in_rew_coef"]
        self.last_avg_ex_rew = learn_info["last_avg_ex_rew"]
        self.log_update_data(data, learn_info)

    def log_update_data(self, data, losses):
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, self.env_step)


def onpolicy_trainer(*args, **kwargs):
    return OnpolicyTrainer(*args, **kwargs).run()
