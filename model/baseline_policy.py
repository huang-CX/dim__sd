import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Type, Union
from tianshou.policy import PPOPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, Collector


class BaselinePolicy(PPOPolicy):
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
        super(BaselinePolicy, self).__init__(**kwargs)
        self.dynamic = dynamic
        self.universal_buffer = universal_buffer
        self.dynamic_optim = dynamic_optim
        self.adapt = adapt
        self.context_size = int(np.prod(getattr(self.actor, 'context_size')))
        self.state_dim = int(np.prod(getattr(self.actor, 'state_shape')))
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

    def train(self, mode: bool = True) -> "BaselinePolicy":
        super(BaselinePolicy, self).train(mode)
        self.dynamic.train(mode)
        return self

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        self.mse_losses = []
        self.ce_losses = []
        self.context_losses = []
        batch.info.update(Batch(oarc=batch.obs, oarc_next=batch.obs_next))
        self.dynamic.train()
        for i in range(self.dyn_epoch):
            output, h = self.dynamic(batch)
            oarc_next = torch.as_tensor(batch.info.oarc_next, device=self.device, dtype=torch.float32)
            if len(oarc_next.shape) == 2:
                obs_next = oarc_next[:, :self.state_dim]
            else:
                obs_next = oarc_next[:, -1, :self.state_dim]
            rew = torch.as_tensor(batch.rew, device=self.device, dtype=torch.float32).unsqueeze(1)
            done = torch.as_tensor(batch.done, device=self.device, dtype=torch.float32).unsqueeze(1)
            gt = torch.concat([obs_next, rew], dim=-1)
            mse_loss = F.mse_loss(torch.concat([output[:, :self.state_dim], output[:, -2:-1]], dim=-1), gt).mean()
            ce_loss = F.cross_entropy(F.sigmoid(output[:, -1:]), done).mean()
            loss = mse_loss + ce_loss
            if mse_loss <= 0.1 and self.augmentation_ratio:
                self.augmentation_flag = True
            self.dynamic_optim.zero_grad()
            loss.backward()
            self.dynamic_optim.step()
            self.mse_losses.append(mse_loss.item())
            self.ce_losses.append(ce_loss.item())
        batch.obs = batch.obs[:, -1, :self.state_dim]
        batch.obs_next = batch.obs_next[:, -1, :self.state_dim]
        # data augmentation
        if self.adapt:
            if self.augmentation_ratio and self.augmentation_flag:
                with torch.no_grad():
                    batch, indices = self.dynamic.process_without_context(batch, indices, self.augmentation_ratio)
                    # batch, indices = self.imagine_trajectory(batch, indices, self.augmentation_ratio)
        batch = super(BaselinePolicy, self).process_fn(batch, buffer, indices)
        return batch

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        res = super(BaselinePolicy, self).learn(batch, batch_size, repeat, **kwargs)
        res.update(
            {
                "loss/augmentation_flag": self.augmentation_flag,
                "loss/mse_loss": self.mse_losses,
                "loss/ce_loss": self.ce_losses,
            }
        )
        return res

    def post_process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        self.augmentation_flag = False
        super(BaselinePolicy, self).post_process_fn(batch, buffer, indices)

    def imagine_trajectory(self, batch: Batch, indices, augmentation_ratio: int = 1, traj_len=10):
        data = batch[:]
        # data.info.update(Batch(oarc=data.obs, oarc_next=data.obs_next))
        state = None
        update_obs = batch.obs
        update_act = batch.act
        update_rew = batch.rew
        update_obs_next = batch.obs_next
        update_done = batch.done
        update_indices = indices
        for i in range(0, augmentation_ratio):
            self.dynamic.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                output, state = self.dynamic(data, state)
                done = F.sigmoid(output[:, -1]).detach().cpu().numpy()
                output = output.cpu().numpy()
                action = self(data).act.cpu().numpy()
                obs_next = output[:, :self.state_dim]
                oarc = np.concatenate([obs_next, action, output[:, self.state_dim].reshape(-1, 1)], axis=-1)
                data.info.oarc = np.delete(data.info.oarc, 0, axis=1)
                data.info.oarc = np.insert(data.info.oarc, 7, oarc.reshape(-1, 1, 26), axis=1)
                data.act = action
                update_obs = np.concatenate([update_obs, data.obs], axis=0)
                update_act = np.concatenate([update_act, data.act], axis=0)
                update_rew = np.concatenate([update_rew, output[:, self.state_dim]], axis=0)
                update_obs_next = np.concatenate(
                    [update_obs_next, obs_next], axis=0)
                update_done = np.concatenate([update_done, done], axis=0)
                update_indices = np.concatenate([update_indices, indices + augmentation_ratio], axis=0)
                data.obs = obs_next
        return Batch(obs=update_obs, act=update_act, rew=update_rew, obs_next=update_obs_next,
                     done=update_done, policy=Batch(), info=Batch()), update_indices
