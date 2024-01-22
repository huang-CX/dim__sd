import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Type, Union
from tianshou.policy import PPOPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, Collector


class MyPolicy(PPOPolicy):
    def __init__(
            self,
            dynamic: nn.Module,
            dynamic_optim: torch.optim.Optimizer,
            universal_buffer: ReplayBuffer = None,
            adapt: Type = False,
            augmentation_ratio: int = 1,
            dyn_batch_size: int = 128,
            dyn_epoch: int = 500,
            **kwargs: Any,
    ) -> None:
        super(MyPolicy, self).__init__(**kwargs)
        self.dynamic = dynamic
        self.universal_buffer = universal_buffer
        self.dynamic_optim = dynamic_optim
        self.adapt = adapt
        self.context_size = int(np.prod(getattr(self.actor, 'context_size')))
        self.state_dim = int(np.prod(getattr(self.actor, 'state_shape'))) - self.context_size
        self.device = getattr(self.actor, 'device')
        self.augmentation_ratio = augmentation_ratio
        self.augmentation_flag = False
        self.mse_losses = []
        self.ce_losses = []
        self.context_losses = []
        self.adapt_context = None
        self.train_mode = False
        self.dyn_batch_size = dyn_batch_size
        self.dyn_epoch = dyn_epoch

    def train(self, mode: bool = True) -> "MyPolicy":
        super(MyPolicy, self).train(mode)
        self.dynamic.train(mode)
        return self

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        self.train_mode = True
        state_dim = int(np.prod(getattr(self.actor, 'state_shape')))
        context_size = int(np.prod(getattr(self.actor, 'context_size')))
        # self.universal_buffer.update(buffer)
        oarc = Batch(oarc=batch.obs, oarc_next=batch.obs_next)
        batch.info.update(Batch(oarc=batch.obs, oarc_next=batch.obs_next))
        self.mse_losses = []
        self.ce_losses = []
        self.context_losses = []
        for i in range(self.dyn_epoch):
            # for b in batch.split(size=self.dyn_batch_size, merge_last=True):
            output, h = self.dynamic(batch)
            oarc_next = torch.as_tensor(batch.info.oarc_next, device=self.device, dtype=torch.float32)
            if len(oarc_next.shape) == 2:
                obs_next = oarc_next[:, :self.state_dim]
            else:
                obs_next = oarc_next[:, -1, :self.state_dim]
            rew = torch.as_tensor(batch.rew, device=self.device, dtype=torch.float32).unsqueeze(1)
            done = torch.as_tensor(batch.done, device=self.device, dtype=torch.float32).unsqueeze(1)
            context = torch.as_tensor(batch.info.oarc_next[:, -1, -self.context_size:],
                                      device=self.device, dtype=torch.float32)
            if self.adapt:
                gt = torch.concat([obs_next, rew], dim=-1)
                mse_loss = F.mse_loss(torch.concat([output[:, :self.state_dim], output[:, -2:-1]], dim=-1), gt).mean()
                ce_loss = F.cross_entropy(F.sigmoid(output[:, -1:]), done).mean()
                context_loss = F.mse_loss(output[:, self.state_dim:self.state_dim + self.context_size], context).mean()
                loss = mse_loss + ce_loss
            else:
                gt = torch.concat([obs_next, rew], dim=-1)
                mse_loss = F.mse_loss(torch.concat([output[:, :self.state_dim], output[:, -2:-1]], dim=-1), gt).mean()
                context_loss = F.mse_loss(output[:, self.state_dim:self.state_dim + self.context_size], context).mean()
                ce_loss = F.cross_entropy(F.sigmoid(output[:, -1:]), done).mean()
                loss = mse_loss + ce_loss + context_loss
            if mse_loss <= 0.1 and self.augmentation_ratio:
                self.augmentation_flag = True
            self.dynamic_optim.zero_grad()
            loss.backward()
            self.dynamic_optim.step()
            self.mse_losses.append(mse_loss.item())
            self.ce_losses.append(ce_loss.item())
            self.context_losses.append(context_loss.item())
        batch.obs = np.concatenate([batch.obs[:, -1, :state_dim - context_size], batch.obs[:, -1, -context_size:]],
                                   axis=-1)
        batch.obs_next = np.concatenate(
            [batch.obs_next[:, -1, :state_dim - context_size], batch.obs_next[:, -1, -context_size:]], axis=-1)
        # data augmentation
        if self.adapt:
            if self.augmentation_ratio and self.augmentation_flag:
                with torch.no_grad():
                    batch = self.dynamic.process(batch, self.augmentation_ratio)
                    indices = np.repeat(indices, self.augmentation_ratio + 1, axis=0)
        batch = super(MyPolicy, self).process_fn(batch, buffer, indices)
        return batch

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        res = super(MyPolicy, self).learn(batch, batch_size, repeat, **kwargs)
        res.update(
            {
                "loss/augmentation_flag": self.augmentation_flag,
                "loss/mse_loss": self.mse_losses,
                "loss/ce_loss": self.ce_losses,
                "loss/context_loss": self.context_losses,
            }
        )
        return res

    def post_process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        # if not self.adapt:
        #     batch.obs, batch.obs_next = batch.info.oarc, batch.info.oarc_next
        self.augmentation_flag = False
        super(MyPolicy, self).post_process_fn(batch, buffer, indices)

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        if self.adapt and not self.train_mode:
            data = Batch()
            data.update(batch)
            if hasattr(data.info, 'oarc'):
                with torch.no_grad():
                    output, _ = self.dynamic(data)
            else:
                data.info.update(Batch(oarc=data.obs, oarc_next=data.obs_next))
                data.update(act=data.obs[:, self.state_dim:self.state_dim + self.actor.output_dim])
                with torch.no_grad():
                    output, state = self.dynamic(data, state)
                pred_context = output[:, self.state_dim:(self.state_dim + self.context_size)].detach().cpu().numpy()
                if hasattr(data.info, 'env_id'):
                    obs = np.concatenate([batch.obs[:, :-self.context_size], self.adapt_context[data.info.env_id]], axis=-1)
                else:
                    obs = np.concatenate([batch.obs[:, :-self.context_size], self.adapt_context], axis=-1)
                batch.update(obs=obs)
        batch = super(MyPolicy, self).forward(batch, state, **kwargs)
        return batch

    def predict_dynamic(self, collector: Collector):
        self.adapt = False
        collector.reset()
        result = collector.collect(n_episode=10)
        data = collector.buffer.sample(0)[0]
        data.info.update(Batch(oarc=data.obs, oarc_next=data.obs_next))
        with torch.no_grad():
            output, _ = self.dynamic(data)
        pred_context = output[:, self.state_dim:(self.state_dim + self.context_size)].detach().cpu().numpy()
        self.adapt_context = np.array([pred_context[data.info.env_id[:, 0] == i].mean(axis=0) for i in collector.env.ready_id])
        self.adapt = True
