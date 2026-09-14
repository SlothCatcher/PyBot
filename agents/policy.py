import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import N_FEATURES


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[512, 256, 128],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = self._mask

        no_valid_action = mask.sum(dim=-1) == 0
        if no_valid_action.any():
            print(f"WARNING: empty action mask for {no_valid_action.sum().item()} batch element(s)")
            mask = mask.clone()
            mask[no_valid_action] = 1

        additive_mask = torch.where(mask == 1, 0.0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + additive_mask)


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]