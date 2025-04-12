"""
    The file contains the PPO class to train with.
    NOTE: Original PPO pseudocode can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

from .base_alg import BasePolicyGradient


class PPO(BasePolicyGradient):
    """
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        super().__init__(policy_class, env, **hyperparameters)
        # TODO: Initialize critic network
        self.critic = policy_class(self.obs_dim, 1)
        # TODO: Initialize optimizers critic
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        
        while t_so_far < total_timesteps:
            # We're collecting our batch simulations here
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            # Increment the number of iterations
            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # TODO: Calculate advantage
            # Calculate advantage using the critic's estimates.
            # First, query the critic to get baseline value estimates for the batch of observations.
            with torch.no_grad():
                V_old = self.critic(batch_obs).squeeze()
            A_k = batch_rtgs - V_old

            # One of the only tricks we use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # TODO: figure out the actor and the critic losses for our algorithm
                V_new, new_log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)
                # Compute the ratio of new and old policy probabilities
                ratios = torch.exp(new_log_probs - batch_log_probs)
                # Compute the surrogate loss terms
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = -torch.min(surr1, surr2).mean()

                # Compute the critic loss as the mean squared error between the new value estimates and the rewards-to-go
                critic_loss = nn.MSELoss()(V_new, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def evaluate(self, batch_obs, batch_acts, batch_rtgs):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # TODO: Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs
