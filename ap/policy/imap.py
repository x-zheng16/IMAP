import torch
from pykeops.torch import Vi, Vj
from torch import nn

from ap.policy.ppo import PPOPolicy

pbe_fn = lambda x, y, k: ((Vi(x) - Vj(y)) ** 2).sum().Kmin(k, 1)  # noqa: E731


class PBE(nn.Module):
    # particle-based estimator
    def __init__(self, k=10, sa_ent=False, style="log_mean"):
        super().__init__()
        self.k = k
        self.sa_ent = sa_ent
        self.style = style

    def forward(self, batch):
        x, y = self.get_xy(batch)
        return self.get_rew(x, y)

    def get_xy(self, batch):
        x = self.get_x(batch)
        x = torch.cat([x, batch.act], -1) if self.sa_ent else x
        return x, x

    def get_x(self, batch):
        x = batch.obs
        x = x[:, None] if x.ndim == 1 else x
        return x

    def get_rew(self, x, y):
        r = pbe_fn(x, y, self.k + 1)[:, 1:].sqrt()
        if self.style == "log_mean":
            rew = torch.log(1 + r.mean(-1))
        elif self.style == "mean":
            rew = r.mean(-1)
        else:
            raise Exception(f"estimation style {self.style} is not supported")
        return rew


class CachePBE(PBE):
    def __init__(self, max_size=int(1e6), **kwargs):
        super().__init__(**kwargs)
        assert self.sa_ent is False
        self.size = max_size
        self.ptr = self.max_ptr = 0
        self.buf = None

    def forward(self, batch, use_cache=True):
        x, y = self.get_xy(batch, use_cache)
        return self.get_rew(x, y)

    def get_xy(self, batch, use_cache=True):
        x = self.get_x(batch)
        self.update_buffer(x)
        # get y
        y = self.buf[: self.n_samples] if use_cache else x
        return x, y

    def update_buffer(self, x):
        B = len(x)
        if self.buf is None:
            self.buf = torch.zeros(self.size, x.shape[1], device=x.device)
        if self.ptr + B > self.size:
            sub = self.size - self.ptr
            self.buf[self.ptr :] = x[:sub]
            self.buf[: B - sub] = x[sub:]
            self.ptr = B - sub
            self.n_samples = self.size
        else:
            self.buf[self.ptr : self.ptr + B] = x
            self.ptr += B
            self.n_samples = self.ptr


class IMAPPPOPolicy(PPOPolicy):
    def __init__(
        self,
        k=10,
        cache_size=int(1e4),
        sa_ent=False,
        style="log_mean",
        use_cache=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pbe = CachePBE(max_size=cache_size, k=k, sa_ent=sa_ent, style=style).to(
            self.device
        )
        self._set_critic("imap")
        self.use_cache = use_cache

    def _get_intrinsic_rew(self, batch, **kwargs):
        rew_imap = self.pbe(batch, self.use_cache)
        batch.info["rew_imap"] = self._to_numpy(rew_imap)
        self.learn_info["rew_intrinsic"] = batch.info["rew_imap"].mean()

    def _get_adv(self, batch, idx):
        return super()._get_adv(batch, idx, "imap")
