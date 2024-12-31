

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
import copy


class WD3QNE_Net(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(WD3QNE_Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def get_q_values(self, state):
        with torch.no_grad():
            return self.forward(state)


class WD3QNE:
    def __init__(self, state_dim=37, num_actions=25, ensemble_size=5, gamma=0.99, tau=0.1):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_size = ensemble_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # Ensemble Q-Networks
        self.Q_ensemble = [WD3QNE_Net(state_dim, num_actions).to(
            self.device) for _ in range(ensemble_size)]
        self.Q_target_ensemble = [copy.deepcopy(
            q_net) for q_net in self.Q_ensemble]

        # Optimizers with AdamW
        self.optimizers = [torch.optim.AdamW(
            q_net.parameters(), lr=0.00005) for q_net in self.Q_ensemble]

    def train(self, batches, epoch):
        (state, next_state, action, next_action,
         reward, done, bloc_num, SOFAS) = batches

        # Normalize state
        state = torch.tensor(state).float().to(self.device)
        next_state = torch.tensor(next_state).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        bloc_num = torch.tensor(bloc_num).long().to(self.device)

        batch_size = 128
        uids = torch.unique(bloc_num)
        num_batches = len(uids) // batch_size

        record_loss = []
        total_loss = 0

        for batch_idx in range(num_batches + 1):
            batch_uids = uids[batch_idx *
                              batch_size: (batch_idx + 1) * batch_size]
            batch_mask = torch.isin(bloc_num, batch_uids)

            # Create batch tensors
            batch = (
                state[batch_mask],
                next_state[batch_mask],
                action[batch_mask],
                reward[batch_mask],
                done[batch_mask],
            )

            loss = self.compute_loss(batch)
            total_loss += loss

            # Log progress
            if batch_idx % 25 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Average Loss: {avg_loss:.4f}")
                record_loss.append(avg_loss)

            # Update target network less frequently
            if batch_idx % 50 == 0:
                self.polyak_target_update()

        return record_loss

    def compute_loss(self, batch):
        state, next_state, action, reward, done = batch
        end_multiplier = 1 - done
        batch_size = state.size(0)
        range_batch = torch.arange(batch_size, device=self.device)

        total_loss = 0
        for q_net, q_target_net, optimizer in zip(self.Q_ensemble, self.Q_target_ensemble, self.optimizers):
            optimizer.zero_grad()

            # Current Q-values
            q_values = q_net(state)
            q_value = q_values[range_batch, action]

            # Next state Double Q-learning
            with torch.no_grad():
                q_values_next = torch.stack(
                    [net.get_q_values(next_state) for net in self.Q_ensemble]).mean(0)
                next_actions = q_values_next.argmax(dim=1)
                q_target_value = q_target_net(
                    next_state)[range_batch, next_actions]

            # Compute TD target with reward scaling
            target_q = reward + self.gamma * q_target_value * end_multiplier
            loss = F.smooth_l1_loss(q_value, target_q)

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / self.ensemble_size

    def polyak_target_update(self):
        with torch.no_grad():
            for q_net, q_target_net in zip(self.Q_ensemble, self.Q_target_ensemble):
                for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
                    target_param.data.mul_(
                        1 - self.tau).add_(self.tau * param.data)

    def get_action(self, state):
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            q_values = torch.stack([net.get_q_values(state)
                                   for net in self.Q_ensemble]).mean(0)
            return torch.argmax(q_values, dim=1).cpu().numpy()

