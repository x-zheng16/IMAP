import json
import os

from omegaconf import OmegaConf
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard.writer import SummaryWriter


class Logger(TensorboardLogger):
    def __init__(self, writer, train_interval=1, update_interval=1, **kwargs):
        super().__init__(
            writer,
            train_interval=train_interval,
            update_interval=update_interval,
            **kwargs,
        )

    def write(self, scope, data, step):
        for k, v in data.items():
            self.writer.add_scalar(f"{scope}/{k}", v, step)
        if self.write_flush:  # issue 580
            self.writer.flush()  # issue #482

    def save_data(self, epoch, env_step, gradient_step, save_checkpoint_fn=None):
        if save_checkpoint_fn:
            save_checkpoint_fn()

    def log_train_data(self, collect_result, step):
        if "game_result" in collect_result:
            for k, v in collect_result["game_result"].items():
                self.writer.add_scalar(k, v, step)
        self.log("train", collect_result, step)

    def log_test_data(self, collect_result, step):
        self.log("test", collect_result, step)

    def log_update_data(self, update_result, step):
        self.write("update", update_result, step)

    def log_info(self, batch, step):
        pass

    def log(self, scope, collect_result, step):
        log_data = {
            "episode": collect_result["n/ep"],
            "reward": collect_result["rew"],
            "length": collect_result["len"],
            "rew": collect_result["rew"],
            "len": collect_result["len"],
            "success_rate": collect_result["success_rate"],
            "fps": collect_result["fps"],
        }
        self.write(scope, log_data, step)


def set_logger(cfg):
    writer = SummaryWriter(".")
    logger = Logger(writer)
    cfg_dict = OmegaConf.to_container(OmegaConf.structured(cfg))
    with open(os.path.join(logger.writer.log_dir, cfg.config_file), "w") as f:
        json.dump(cfg_dict, f, indent=4)
    return logger
