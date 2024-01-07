# Multi-agent as vectorized environment
import torch
from torch import nn

from box import Box 
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from utils.sb3.reg_ppo import RegularizedPPO


class LateFusionAttn(nn.Module):
    """
    Custom network with attention for policy and value function. 
    
    Args:
        feature_dim (int): dimension of the input features
        arch_ego_state (List[int]): list of layer dimensions for the ego state network
        arch_road_objects (List[int]): list of layer dimensions for the road objects network
        arch_road_graph (List[int]): list of layer dimensions for the road graph network
        arch_shared (List[int]): list of output layer dimensions for the shared network
        act_func (str): activation function for the network
        last_layer_dim_pi (int): dimension of the output layer for the policy network
        last_layer_dim_vf (int): dimension of the output layer for the value network
    """

    def __init__(
        self,
        feature_dim: int,
        env_config: Box,
        arch_ego_state: List[int] = [8],
        arch_road_objects: List[int] = [32],
        arch_road_graph: List[int] = [32],
        arch_stop_sign: List[int] = [4],
        arch_shared: List[int] = [128],
        act_func: str = "tanh", 
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = env_config
        self.arch_ego_state = arch_ego_state
        self.arch_road_objects = arch_road_objects
        self.arch_road_graph = arch_road_graph
        self.arch_shared = arch_shared  
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.dropout = dropout

        # Define original object dimensions
        self.ro_feat, self.ro_max = 13, self.config.scenario.max_visible_objects
        self.rg_feat, self.rg_max = 13, self.config.scenario.max_visible_road_points
        self.tl_feat, self.tl_max = 12, self.config.scenario.max_visible_traffic_lights
        self.ss_feat, self.ss_max = 3, self.config.scenario.max_visible_stop_signs

        shared_input_dim = (
            arch_ego_state[-1] \
            +  self.ro_max * arch_road_objects[-1] \
            +  self.rg_max * arch_road_graph[-1] \
            +  self.ss_max * arch_stop_sign[-1]
        )

        # IMPORTANT:Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Get scene modality dimensions
        self.input_dim_ego, self.input_dim_road_objects, self.input_dim_stop_signs, self.input_dim_road_graph = self._get_obs_input_dims()
    
        # # # # # POLICY NETWORK # # # # #
        # EGO STATE
        self.policy_net_ego_state = self._build_ego_state_net(arch_ego_state)
        # STOP SIGNS
        self.policy_net_stop_signs = self._build_stop_sign_net(arch_stop_sign)
        # ROAD OBJECTS
        self.policy_net_road_objects = self._build_road_objects_net(arch_road_objects)
        #self.policy_net_ro_attn = nn.MultiheadAttention(embed_dim=arch_road_objects[-1], num_heads=1, batch_first=True)
        # Flatten the road objects
        #self.policy_net_ro_norm1 = nn.LayerNorm(arch_road_graph[-1])
        # ROAD GRAPH
        self.policy_net_road_graph = self._build_road_graph_net(arch_road_graph)
        #self.policy_net_rg_attn = nn.MultiheadAttention(embed_dim=arch_road_graph[-1], num_heads=1, batch_first=True)
        #self.policy_net_rg_norm1 = nn.LayerNorm(arch_road_graph[-1])
        
        # Fuse and output layer
        self.policy_out_net = nn.Sequential(
            nn.Linear(shared_input_dim, self.latent_dim_pi),
            nn.LayerNorm(self.latent_dim_pi),
            self.act_func,
        )

        # # # # # VALUE NETWORK # # # # #
        # EGO STATE
        self.val_net_ego_state = self._build_ego_state_net(arch_ego_state)
        # STOP SIGNS
        self.val_net_stop_signs = self._build_stop_sign_net(arch_stop_sign)
        # ROAD OBJECTS
        self.val_net_road_objects = self._build_road_objects_net(arch_road_objects)
        #self.val_net_ro_attn = nn.MultiheadAttention(embed_dim=arch_road_objects[-1], num_heads=1, batch_first=True)
        # Flatten the road objects
        #self.val_net_ro_norm1 = nn.LayerNorm(arch_road_graph[-1])
        # ROAD GRAPH
        
        self.val_net_road_graph = self._build_road_graph_net(arch_road_graph)
        #self.val_net_rg_attn = nn.MultiheadAttention(embed_dim=arch_road_graph[-1], num_heads=1, batch_first=True)
        #self.val_net_rg_norm1 = nn.LayerNorm(arch_road_graph[-1])
        
        # Fuse and output layer
        self.val_out_net = nn.Sequential(
            nn.Linear(shared_input_dim, self.latent_dim_pi),
            nn.LayerNorm(self.latent_dim_pi),
            self.act_func,
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): input tensor of shape (batch_size, feature_dim)
        Return:
            (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the policy network."""
        
        # Unflatten the obs to get the ego state and the visible state items
        ego_state, road_objects, stop_signs, road_points = self._unflatten_obs(features)    
        
        # Embeddings
        ego_state = self.policy_net_ego_state(ego_state)
        road_objects = self.policy_net_road_objects(road_objects)
        road_points = self.policy_net_road_graph(road_points)
        stop_signs = self.policy_net_stop_signs(stop_signs)

        # Attention
        # road_objects, _ = self.policy_net_ro_attn(road_objects, road_objects, road_objects)
        # road_points, _ = self.policy_net_rg_attn(road_points, road_points, road_points)

        # # Layer norm
        # road_objects = self.policy_net_ro_norm1(road_objects)
        # road_points = self.policy_net_rg_norm1(road_points)

        # Flatten: (N, E) -> (N * E)
        road_objects = road_objects.flatten(start_dim=1)
        road_points = road_points.flatten(start_dim=1)
        stop_signs = stop_signs.flatten(start_dim=1)

        # Merge and FFN 
        policy_out = self.policy_out_net(torch.cat((ego_state, road_objects, road_points, stop_signs), dim=1))
        
        return policy_out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""
         # Unflatten the obs to get the ego state and the visible state items
        ego_state, road_objects, stop_signs, road_points = self._unflatten_obs(features)    
        
        # Embeddings
        ego_state = self.val_net_ego_state(ego_state)
        road_objects = self.val_net_road_objects(road_objects)
        road_points = self.val_net_road_graph(road_points)
        stop_signs = self.val_net_stop_signs(stop_signs)

        # Attention
        # road_objects, _ = self.policy_net_ro_attn(road_objects, road_objects, road_objects)
        # road_points, _ = self.policy_net_rg_attn(road_points, road_points, road_points)

        # # Layer norm
        # road_objects = self.policy_net_ro_norm1(road_objects)
        # road_points = self.policy_net_rg_norm1(road_points)

        # Flatten: (N, E) -> (N * E)
        road_objects = road_objects.flatten(start_dim=1)
        road_points = road_points.flatten(start_dim=1)
        stop_signs = stop_signs.flatten(start_dim=1)

        # Merge and FFN 
        val_out = self.policy_out_net(torch.cat((ego_state, road_objects, road_points, stop_signs), dim=1))
        
        return val_out

    def _build_ego_state_net(self, net_arch: List[int]):
        """Create ego state network architecture."""
        net_layers = []
        prev_dim = self.input_dim_ego
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net
    
    def _build_stop_sign_net(self, net_arch: List[int]):
        """Create traffic objects network architecture."""
        net_layers = []
        prev_dim = self.input_dim_stop_signs
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net
    
    def _build_road_objects_net(self, net_arch: List[int]):
        """Create road objects architecture."""
        net_layers = []
        prev_dim = self.input_dim_road_objects 
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net

    def _build_road_graph_net(self, net_arch: List[int]):
        """Create road graph network architecture."""
        net_layers = [] 
        prev_dim = self.input_dim_road_graph 
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net

    def _build_out_net(self, input_dim: int, output_dim: int, net_arch: List[int]):
        """Create the output network architecture."""
        net_layers = [] 
        prev_dim = input_dim
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            net_layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim
        # Add final layer
        net_layers.append(nn.Linear(prev_dim, output_dim))
        net_layers.append(nn.LayerNorm(output_dim))
        net_layers.append(self.act_func)

        net = nn.Sequential(*net_layers)    
        return net
    
    def _unflatten_obs(self, obs_flat):
        """Recover indivdiual scenario information from a flattened observation tensor.
        
        Args:
            obs_flat (torch.Tensor): flattened observation tensor
            dim_ego (int): dimension of the ego state

        Returns:
            ego_state (torch.Tensor): ego state tensor
            road_objects (torch.Tensor): road objects tensor
            road_points (torch.Tensor): road points tensor
        """

        # Get ego and visible state
        ego_state, vis_state = obs_flat[:, :self.input_dim_ego], obs_flat[:, self.input_dim_ego:]

        # Visible state object order: road_objects, road_points, traffic_lights, stop_signs
        # Find the ends of each section
        ROAD_OBJECTS_END = self.ro_feat * self.ro_max
        ROAD_POINTS_END = ROAD_OBJECTS_END + (self.rg_feat * self.rg_max)
        TL_END = ROAD_POINTS_END + (self.tl_feat * self.tl_max)
        STOP_SIGN_END = TL_END + (self.ss_feat * self.ss_max)
        
        # Unflatten and reshape to (batch_size, num_objects, object_dim)
        road_objects = (vis_state[:, :ROAD_OBJECTS_END]).reshape(-1, self.ro_max, self.ro_feat)
        road_points = (vis_state[:, ROAD_OBJECTS_END:ROAD_POINTS_END]).reshape(-1, self.rg_max, self.rg_feat)
        stop_signs = (vis_state[:, TL_END:STOP_SIGN_END]).reshape(-1, self.ss_max, self.ss_feat)    
        
        # Ommit traffic lights for now
        traffic_lights = (vis_state[:, ROAD_POINTS_END:TL_END])
      
        return ego_state, road_objects, stop_signs, road_points
    
    def _get_obs_input_dims(self):
        """Get the input dimensions for the ego state, road objects, and road graph."""
        ego_state_dim = 10
        road_objects_dim = 13
        stop_sign_dim = 3
        road_graph_dim = 13 
        return ego_state_dim, road_objects_dim, stop_sign_dim, road_graph_dim


class LateFusionAttnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # Build the network architecture
        self.mlp_extractor = LateFusionAttn(
            self.features_dim, 
            env_config
        )

if __name__ == "__main__":

    # Load configs
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    
    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
    )

    obs = env.reset()
    obs = torch.Tensor(obs)[:2]

    # Test
    model = RegularizedPPO(
        reg_policy=None,
        reg_weight=None, # Regularization weight; lambda
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=LateFusionAttnPolicy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        verbose=1,
        device='cuda',
    )
    # print(model.policy)
    model.learn(5000)