
import wandb

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

# Utils
from utils.config import load_config

# Imitation learning data iterator
from utils.imitation_learning.waymo_iterator import TrajectoryIterator

if __name__ == "__main__":
    
    BATCH_SIZE = 512
    FILE_LIMIT = 5
    EPOCHS = 800
    MINIBATCHES = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configs
    bc_config = load_config("bc_config")
    env_config = load_config("env_config")
    
    # Run 
    run = wandb.init(
        project="bc_from_scratch",
    )

    # Prepare dataset
    waymo_iterator = TrajectoryIterator(
        env_config=env_config,
        data_path=env_config.data_path,
        apply_obs_correction=False,
        file_limit=env_config.num_files,
    )
    
    data_loader = iter(
        DataLoader(
            waymo_iterator,
            batch_size=BATCH_SIZE,
            pin_memory=True,
        )
    )
    
    # Define network
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Net, self).__init__()
            self.nn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])
        
        def dist(self, obs):
            """Generate action distribution."""
            x_out = self.nn(obs.float())
            return [Categorical(logits=head(x_out)) for head in self.heads]
            
        def forward(self, obs, deterministic=False):
            """Generate an output from tensor input."""
            action_dist = self.dist(obs)
            
            if deterministic:
                actions_idx = action_dist[0].logits.argmax(axis=-1) 
            else:
                actions_idx = action_dist.sample()
            return actions_idx
        
        def _log_prob(self, obs, expert_actions):
            pred_action_dist = self.dist(obs)
            log_prob = pred_action_dist[0].log_prob(expert_actions).mean()
            return log_prob
            
    # Build model
    bc_policy = Net(
        input_size=waymo_iterator.observation_space.shape[0], 
        hidden_size=800, 
        output_size=waymo_iterator.action_space.n
    ).to(DEVICE)
    
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=1e-4)
    
    global_step = 0
    for epoch in range(EPOCHS):    
        for i in range(MINIBATCHES): 
          
            # Get batch of obs-act pairs
            obs, expert_action, _, _ = next(data_loader)
            obs, expert_action = obs.to(DEVICE), expert_action.to(DEVICE)
            
            # Forward pass
            log_prob = bc_policy._log_prob(obs, expert_action.float())
            loss = -log_prob
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            with torch.no_grad():
                pred_action = bc_policy(obs, deterministic=True)
                accuracy = (expert_action == pred_action).sum() / expert_action.shape[0]        
                
            wandb.log({
                "global_step": global_step,
                "loss": loss.item(),
                "acc": accuracy,
            })
            
            global_step += 1
