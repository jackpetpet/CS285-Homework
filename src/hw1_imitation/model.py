"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim,chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        pred = self.model(state)
        pred = pred.view(B,self.chunk_size,self.action_dim)
        return F.mse_loss(pred,action_chunk)


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        pred = self.model(state)
        return pred.view(B,self.chunk_size,self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        input_dim = state_dim+(chunk_size*action_dim)+1

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim,chunk_size*action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand((B,1,1),device=state.device)
        A_tau = tau*action_chunk + (1-tau)*noise
        A_tau_flat = A_tau.view(B,-1)
        tau_flat = tau.view(B,-1)
        net_input = torch.cat([state,A_tau_flat,tau_flat],dim=-1)
        pred_velocity = self.model(net_input).view(B,self.chunk_size,self.action_dim)
        target_velocity = action_chunk - noise
        return F.mse_loss(pred_velocity,target_velocity)



    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        current_action = torch.randn((B,self.chunk_size,self.action_dim),device=state.device)
        step_size = 1.0 / num_steps
        tau = 0.0
        for _ in range(num_steps):
            current_action_flat = current_action.view(B,-1)
            tau_tensor = torch.full((B,1),tau,device=state.device)
            net_input = torch.cat([state,current_action_flat,tau_tensor],dim=-1)
            velocity = self.model(net_input).view(B,self.chunk_size,self.action_dim)
            current_action = current_action + step_size * velocity
            tau += step_size

        return current_action



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
