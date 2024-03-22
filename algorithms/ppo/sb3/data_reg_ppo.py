"""Module containing regularized PPO algorithm."""

import logging
import io
import pathlib
import numpy as np
from stable_baselines3.common.utils import explained_variance
import torch
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from algorithms.ppo.sb3.ma_ppo import MultiAgentPPO
from utils.imitation_learning.waymo_iterator import TrajectoryIterator
# Torch
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.getLogger(__name__)

class DataRegularizedPPO(MultiAgentPPO):
    """Regularized PPO that is compatible with multi-agent environments.
    Args:
        reg_policy (stable_baselines3.common.policies.ActorCriticPolicy): Regularization policy.
        reg_weight (float): Weight of regularization loss.
        reg_loss (torch.nn.Module): Regularization loss function.
    """
    def __init__(
        self,
        reg_weight=None,
        *,
        reg_loss=None,
        expert_data_batch_size=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if reg_weight is None:
            logging.info("No regularization weight specified, using default PPO.")
        elif not isinstance(reg_weight, float):
            raise TypeError(
                f"reg_weight must be float between 0.0 and 1.0, got {type(reg_weight)}."
            )
        elif not (0.0 <= reg_weight <= 1.0):
            raise ValueError(
                f"reg_weight must be float between 0.0 and 1.0, got {reg_weight}."
            )
        self.reg_weight = reg_weight
        
        # Expert dataset and loader
        self.waymo_iterator = TrajectoryIterator(
            env_config=self.env_config,
            data_path=self.env_config.data_path,
            apply_obs_correction=False,
            file_limit=self.env_config.num_files,
        )
        
        self.expert_data_loader = iter(
            DataLoader(
                self.waymo_iterator,
                batch_size=expert_data_batch_size,
                pin_memory=True,
            )
        )
   
    def eval_rl_policy_on_expert_data(self):
        """Compute the negative log likelihood (NLL) of the expert actions under the RL policy."""
        
        # Get batch of obs-expert act pairs
        observations, expert_actions, _, _ = next(self.expert_data_loader)
        observations, expert_actions = observations.to(self.device), expert_actions.to(self.device)
        
        # Calculate the log likelihood of the expert actions under to the RL policy
        log_prob = self.policy.get_distribution(
            observations).log_prob(expert_actions).mean()
        
        # Compute loss
        loss = -log_prob
        
        return loss

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                
                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                loss_ppo = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Compute loss on expert data
                nll_expert_actions = self.eval_rl_policy_on_expert_data()

                if self.reg_weight is not None:
                    reg_weight = self.reg_weight
                    
                    # Define the loss as a weighted sum of the PPO loss and the regularization loss
                    loss = (1 - reg_weight) * loss_ppo + reg_weight * nll_expert_actions
                else:
                    # Use default PPO loss
                    loss = loss_ppo
                
                # # # # # # # # # HR_PPO EDIT # # # # # # # # #
                
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("regularize/reg_weight", reg_weight)
        self.logger.record("regularize/loss_ppo", np.abs(loss_ppo.item()))
        self.logger.record("regularize/loss_expert_data", nll_expert_actions.item())
        self.logger.record("regularize/loss_expert_data_weighted", reg_weight * nll_expert_actions.item())
        self.logger.record("regularize/loss_ppo_weighted", (1 - reg_weight) * np.abs(loss_ppo.item()))

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            
    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        
        # EDIT: Make sure to exclude the expert data loader and waymo iterator
        exclude = ['waymo_iterator', 'expert_data_loader']

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)