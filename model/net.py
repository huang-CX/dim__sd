import numpy as np
import torch
import torch.nn.functional as F

from typing import Any, Dict, Optional, Sequence, Tuple, Union
from torch import nn
from tianshou.utils.net.common import MLP, Recurrent, Net
from tianshou.data import Batch

SIGMA_MIN = -20
SIGMA_MAX = 2


class MQLRecurrentActor(nn.Module):
    """Simple Recurrent network based on LSTM.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            layer_num: int,
            state_shape: Union[int, Sequence[int]],
            action_shape: Union[int, Sequence[int]],
            device: Union[str, int, torch.device] = "cpu",
            context_hidden_size: int = 128,
            context_size: int = 8,
            hidden_sizes: Sequence[int] = (),
            max_action: float = 1.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=context_hidden_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
        self.hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        self.mu = MLP(context_size + 20, output_dim, hidden_sizes,
                      device=self.device)  # 20 represents the actual state dim
        self._max = max_action

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mapping: s -> flatten -> logits.

        In the evaluation mode, s should be with shape ``[bsz, dim]``; in the
        training mode, s should be with shape ``[bsz, len, dim]``. See the code
        and comment for more detail.
        """
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            _, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            _, (h, c) = self.nn(
                s, (
                    state["h"].transpose(0, 1).contiguous(),
                    state["c"].transpose(0, 1).contiguous()
                )
            )
        context = self.hidden_output(h.squeeze(0))
        s = s[:, -1]
        logits = torch.cat([context, s[:, :20]], dim=-1)  # 20 represents the actual state dim
        mu = self.mu(logits)
        mu = self._max * torch.tanh(mu)
        # please ensure the first dim is batch size: [bsz, len, ...]
        return mu, {"h": h.transpose(0, 1).detach(), "c": c.transpose(0, 1).detach()}


class MQLRecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            layer_num: int,
            state_shape: Sequence[int],
            action_shape: Sequence[int],
            context_hidden_size: int = 128,
            context_size: int = 8,
            hidden_sizes: Sequence[int] = (),
            max_action: float = 1.0,
            device: Union[str, int, torch.device] = "cpu",
            unbounded: bool = False,
            conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=context_hidden_size,
            num_layers=layer_num,
            batch_first=True,
        )
        output_dim = int(np.prod(action_shape))
        self.hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        self.mu = MLP(context_size + 20, output_dim, hidden_sizes,
                      device=self.device)  # 20 represents the actual state dim
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(context_size + int(np.prod(state_shape)), output_dim, hidden_sizes,
                             device=self.device)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Optional[Dict[str, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            _, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            _, (h, c) = self.nn(
                s, (
                    state["h"].transpose(0, 1).contiguous(),
                    state["c"].transpose(0, 1).contiguous()
                )
            )
        context = self.hidden_output(h.squeeze(0))
        s = s[:, -1]
        logits = torch.cat([context, s[:, :20]], dim=-1)  # 20 represents the actual state dim
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {
            "h": h.transpose(0, 1).detach(),
            "c": c.transpose(0, 1).detach()
        }


class MQLRecurrentCritic(nn.Module):
    """Recurrent version of Critic.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
            self,
            layer_num: int,
            state_shape: Sequence[int],
            action_shape: Sequence[int] = [0],
            device: Union[str, int, torch.device] = "cpu",
            context_hidden_size: int = 128,
            context_size: int = 8,
            hidden_sizes: Sequence[int] = ()
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(
            input_size=int(np.prod(state_shape)),
            hidden_size=context_hidden_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        self.q = MLP(20 + int(np.prod(action_shape)) + context_size, 1, hidden_sizes,
                     device=self.device)  # 20 represents the actual state dim

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)  # type: ignore
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        _, (h, c) = self.nn(s)
        s = s[:, -1]
        s = s[:, :20]  # 20 represents the actual state dim
        if a is not None:
            a = torch.as_tensor(
                a,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
            s = torch.cat([s, a], dim=1)
        h = self.hidden_output(h.squeeze(0))
        combined = torch.cat([s, h], dim=1)
        q = self.q(combined)
        return q


class MetaDynamic(nn.Module):
    def __init__(
            self,
            oarc_shape: Sequence[int],
            action_shape: Sequence[int] = [0],
            context_hidden_size: int = 128,
            context_size: int = 7,
            device: Union[str, int, torch.device] = "cpu",
            hidden_sizes: Sequence[int] = ()
    ) -> None:
        super(MetaDynamic, self).__init__()
        self.oarc_shape = oarc_shape
        self.action_shape = action_shape
        self.device = device
        self.context_size = context_size
        self.state_dim = int(np.prod(oarc_shape)) - context_size - int(np.prod(action_shape)) - 1
        input_dim = int(np.prod(oarc_shape)) - context_size
        self.nn = nn.LSTM(
            input_size=input_dim,
            hidden_size=context_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.context_size = context_size
        self.hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        output_dim = int(np.prod(oarc_shape)) - int(np.prod(action_shape)) + 1
        self.model = MLP(int(np.prod(oarc_shape)) - 1, output_dim,
                         hidden_sizes=hidden_sizes, device=self.device)

    def forward(
            self,
            batch: Batch = None,
            state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple]:
        obs_act_rew_context = torch.as_tensor(batch.info.oarc, device=self.device, dtype=torch.float32)
        if len(obs_act_rew_context.shape) == 2:
            obs_act_rew_context = obs_act_rew_context.unsqueeze(-2)
        obs_act_rew = obs_act_rew_context[:, :, :-self.context_size]  # task context dim is 7
        act = torch.as_tensor(batch.act, device=self.device, dtype=torch.float32)
        assert len(obs_act_rew.shape) == 3
        self.nn.flatten_parameters()
        if state is None:
            _, (h, c) = self.nn(obs_act_rew)
        else:
            _, (h, c) = self.nn(
                obs_act_rew, (
                    state["h"].transpose(0, 1).contiguous(),
                    state["c"].transpose(0, 1).contiguous()
                )
            )
        hidden = self.hidden_output(h.squeeze(0))
        obs_act_rew = obs_act_rew[:, -1]
        obs = obs_act_rew[:, :self.state_dim]  # 20 represents the actual state dim
        combined = torch.concat([obs, act, hidden], dim=1)
        output = self.model(combined)
        return output, {"h": h.transpose(0, 1).detach(), "c": c.transpose(0, 1).detach()}

    def process(
            self,
            act,
            batch: Batch = None
    ):
        obs_act_rew_context = torch.as_tensor(batch.info.oarc, device=self.device, dtype=torch.float32)
        obs_act_rew = obs_act_rew_context[:, :, :-self.context_size]  # task context dim is 8
        # obs_next = torch.as_tensor(batch.info.oarc_next[:, -1, :self.state_dim], device=self.device, dtype=torch.float32)
        # context = torch.as_tensor(batch.obs_next[:, -self.context_size:],
        #                           device=self.device, dtype=torch.float32)
        act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
        assert len(obs_act_rew.shape) == 3
        self.nn.flatten_parameters()
        _, (h, c) = self.nn(obs_act_rew)
        h = self.hidden_output(h.squeeze(0))
        obs_act_rew = obs_act_rew[:, -1]
        obs = obs_act_rew[:, :self.state_dim]
        combined = torch.concat([obs, act, h], dim=-1)
        output = self.model(combined)
        update_obs = np.concatenate([batch.obs, batch.obs], axis=0)
        update_act = np.concatenate([batch.act, act.detach().cpu().numpy()], axis=0)
        update_rew = np.concatenate([batch.rew, output[:, self.state_dim + self.context_size].detach().cpu().numpy()],
                                    axis=0)
        update_obs_next = np.concatenate(
            [batch.obs_next, output[:, :self.state_dim + self.context_size].detach().cpu().numpy()], axis=0)
        update_done = np.concatenate([batch.done, F.sigmoid(output[:, -1]).detach().cpu().numpy()], axis=0)
        # update_policy = np.concatenate([batch.policy, batch.policy], axis=0)
        return Batch(obs=update_obs, act=update_act, rew=update_rew, obs_next=update_obs_next,
                     done=update_done, policy=Batch(), info=Batch())


class WorldModel(nn.Module):
    def __init__(
            self,
            oarc_shape: Sequence[int],
            action_shape: Sequence[int] = [0],
            context_hidden_size: int = 128,
            context_size: int = 7,
            device: Union[str, int, torch.device] = "cpu",
            hidden_sizes: Sequence[int] = ()
    ) -> None:
        super(WorldModel, self).__init__()
        self.oarc_shape = oarc_shape
        self.action_shape = action_shape
        self.device = device
        self.context_size = context_size
        self.state_dim = int(np.prod(oarc_shape)) - context_size - int(np.prod(action_shape)) - 1
        input_dim = int(np.prod(oarc_shape)) - context_size
        self.context_nn = nn.LSTM(
            input_size=input_dim,
            hidden_size=context_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.world_model_nn = nn.LSTM(
            input_size=input_dim,
            hidden_size=context_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.context_size = context_size
        self.context_hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        self.world_model_hidden_output = nn.Linear(context_hidden_size, context_size, device=self.device)
        output_dim = int(np.prod(oarc_shape)) - int(np.prod(action_shape)) + 1 - context_size
        self.context_model = MLP(int(np.prod(oarc_shape)) - 1, output_dim,
                                 hidden_sizes=hidden_sizes, device=self.device)

    def forward(
            self,
            batch: Batch = None,
            state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple]:
        obs_act_rew_context = torch.as_tensor(batch.info.oarc, device=self.device, dtype=torch.float32)
        if len(obs_act_rew_context.shape) == 2:
            obs_act_rew_context = obs_act_rew_context.unsqueeze(-2)
        if obs_act_rew_context.shape[-1] == 34:
            obs_act_rew = obs_act_rew_context[:, :, :-self.context_size]  # task context dim is 8
        elif obs_act_rew_context.shape[-1] == 26:
            obs_act_rew = obs_act_rew_context
        else:
            raise Exception("Invalid shape of obs_act_rew")
        context = obs_act_rew_context[:, :, -self.context_size:]
        act = torch.as_tensor(batch.act, device=self.device, dtype=torch.float32)
        assert len(obs_act_rew.shape) == 3
        self.context_nn.flatten_parameters()
        self.world_model_nn.flatten_parameters()
        if state is None:
            _, (context_h, context_c) = self.context_nn(obs_act_rew)
            _, (world_model_h, world_model_c) = self.world_model_nn(obs_act_rew)
        else:
            _, (context_h, context_c) = self.context_nn(
                obs_act_rew, (
                    state["context_h"].transpose(0, 1).contiguous(),
                    state["context_c"].transpose(0, 1).contiguous()
                )
            )
            _, (world_model_h, world_model_c) = self.world_model_nn(
                obs_act_rew, (
                    state["world_model_h"].transpose(0, 1).contiguous(),
                    state["world_model_c"].transpose(0, 1).contiguous()
                )
            )
        context_hidden = self.context_hidden_output(context_h.squeeze(0))
        world_model_pre_context = self.world_model_hidden_output(world_model_h.squeeze(0))
        obs_act_rew = obs_act_rew[:, -1]
        obs = obs_act_rew[:, :self.state_dim]  # 20 represents the actual state dim
        context_combined = torch.concat([obs, act, context_hidden], dim=1)
        output = self.context_model(context_combined)
        output = torch.concat([output[:, :self.state_dim], world_model_pre_context, output[:, self.state_dim:]], dim=-1)
        return output, {"context_h": context_h.transpose(0, 1).detach(), "context_c": context_c.transpose(0, 1).detach(),
                        "world_model_h": world_model_h.transpose(0, 1).detach(),
                        "world_model_c": world_model_c.transpose(0, 1).detach()}

    def process(
            self,
            batch: Batch = None,
            augmentation_ratio: int = 1
    ):
        obs_act_rew_context = torch.as_tensor(batch.info.oarc, device=self.device, dtype=torch.float32)
        obs_act_rew = obs_act_rew_context[:, :, :-self.context_size]  # task context dim is 8
        # obs_next = torch.as_tensor(batch.info.oarc_next[:, -1, :self.state_dim], device=self.device, dtype=torch.float32)
        context = batch.obs_next[:, -self.context_size:]
        # act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
        assert len(obs_act_rew.shape) == 3
        self.context_nn.flatten_parameters()
        _, (h, c) = self.context_nn(obs_act_rew)
        h = self.context_hidden_output(h.squeeze(0))
        obs_act_rew = obs_act_rew[:, -1]
        obs = obs_act_rew[:, :self.state_dim]
        update_obs = batch.obs
        update_act = batch.act
        update_rew = batch.rew
        update_obs_next = batch.obs_next
        update_done = batch.done
        for _ in range(augmentation_ratio):
            act = batch.act + 0.03 * np.random.randn(batch.act.shape[0], batch.act.shape[1])
            act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
            combined = torch.concat([obs, act, h], dim=-1)
            output = self.context_model(combined)
            update_obs = np.concatenate([update_obs, batch.obs], axis=0)
            update_act = np.concatenate([update_act, act.detach().cpu().numpy()], axis=0)
            update_rew = np.concatenate([update_rew, output[:, self.state_dim].detach().cpu().numpy()],
                                        axis=0)
            oc_next = np.concatenate([output[:, :self.state_dim].detach().cpu().numpy(), context], axis=-1)
            update_obs_next = np.concatenate(
                [update_obs_next, oc_next], axis=0)
            update_done = np.concatenate([update_done, F.sigmoid(output[:, -1]).detach().cpu().numpy()], axis=0)
            # update_policy = np.concatenate([batch.policy, batch.policy], axis=0)
        return Batch(obs=update_obs, act=update_act, rew=update_rew, obs_next=update_obs_next,
                     done=update_done, policy=Batch(), info=Batch())

    def process_without_context(
            self,
            batch: Batch = None,
            indices: np.array = None,
            augmentation_ratio: int = 1,
            
    ):
        obs_act_rew_context = torch.as_tensor(batch.info.oarc, device=self.device, dtype=torch.float32)
        obs_act_rew = obs_act_rew_context[:, :, :]  # task context dim is 8
        # obs_next = torch.as_tensor(batch.info.oarc_next[:, -1, :self.state_dim], device=self.device, dtype=torch.float32)
        # context = torch.as_tensor(batch.obs_next[:, -self.context_size:],
        #                           device=self.device, dtype=torch.float32)
        
        assert len(obs_act_rew.shape) == 3
        self.context_nn.flatten_parameters()
        _, (h, c) = self.context_nn(obs_act_rew)
        h = self.context_hidden_output(h.squeeze(0))
        obs_act_rew = obs_act_rew[:, -1]
        obs = obs_act_rew[:, :self.state_dim]
        update_obs = batch.obs
        update_act = batch.act
        update_rew = batch.rew
        update_obs_next = batch.obs_next
        update_done = batch.done
        update_indices = indices
        act = torch.as_tensor(batch.act, device=self.device, dtype=torch.float32)
        output = torch.concat([obs, act, h], dim=-1)
        loss = (output[:, :self.state_dim].detach().cpu().numpy() - batch.obs_next) ** 2
        weight = loss.mean(axis=1)
        index = np.where(weight < 0.1)
        for _ in range(augmentation_ratio):
            act = batch.act + 0.02 * np.random.randn(batch.act.shape[0], batch.act.shape[1])
            act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
            combined = torch.concat([obs[index], act[index], h[index]], dim=-1)
            output = self.context_model(combined)
            update_obs = np.concatenate([update_obs, batch.obs[index]], axis=0)
            update_act = np.concatenate([update_act, act.detach().cpu().numpy()[index]], axis=0)
            update_rew = np.concatenate([update_rew, output[:, self.state_dim].detach().cpu().numpy()],
                                        axis=0)
            update_obs_next = np.concatenate(
                [update_obs_next, output[:, :self.state_dim].detach().cpu().numpy()], axis=0)
            update_done = np.concatenate([update_done, F.sigmoid(output[:, -1]).detach().cpu().numpy()], axis=0)
            update_indices = np.concatenate([update_indices, indices[index] + 1], axis=0)
        # update_policy = np.concatenate([batch.policy, batch.policy], axis=0)
        return Batch(obs=update_obs, act=update_act, rew=update_rew, obs_next=update_obs_next,
                     done=update_done, policy=Batch(), info=Batch()), update_indices


class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            action_shape: Sequence[int],
            state_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            context_size: int = 8,
            max_action: float = 1.0,
            device: Union[str, int, torch.device] = "cpu",
            unbounded: bool = False,
            conditioned_sigma: bool = False,
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.context_size = context_size
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim, self.output_dim, hidden_sizes, device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        if self.context_size > 0:
            s = np.concatenate([s[:, :int(np.prod(self.state_shape)) - self.context_size], s[:, -self.context_size:]],
                               axis=-1)
        else:
            s = s[:, :int(np.prod(self.state_shape))]
        logits, h = self.preprocess(s, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
            self,
            preprocess_net: nn.Module,
            state_shape: Sequence[int],
            hidden_sizes: Sequence[int] = (),
            context_size: int = 8,
            device: Union[str, int, torch.device] = "cpu",
            preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.context_size = context_size
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, 1, hidden_sizes, device=self.device)

    def forward(
            self,
            s: Union[np.ndarray, torch.Tensor],
            a: Optional[Union[np.ndarray, torch.Tensor]] = None,
            info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = torch.as_tensor(
            s,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        if self.context_size > 0:
            s = torch.concat([s[:, :int(np.prod(self.state_shape)) - self.context_size], s[:, -self.context_size:]], dim=-1)
        else:
            s = s[:, :int(np.prod(self.state_shape))]
        if a is not None:
            a = torch.as_tensor(
                a,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
            s = torch.cat([s, a], dim=1)
        logits, h = self.preprocess(s)
        logits = self.last(logits)
        return logits


if __name__ == "__main__":
    dyn = MetaDynamic(state_shape=(20,), action_shape=(5,))
    obs = torch.randn([256, 8, 26])
    act = torch.randn([256, 5])
    delta_s = dyn(obs, act)
    print(delta_s.shape)
