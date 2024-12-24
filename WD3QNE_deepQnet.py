# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim
# import torch.nn.functional as F
# import copy

# device = 'cpu'


# class WD3QNE_Net(nn.Module):
#     def __init__(self, state_dim, n_actions):
#         super(WD3QNE_Net, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#         )
#         self.fc_val = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#         self.fc_adv = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, n_actions)
#         )

#     def forward(self, state):
#         conv_out = self.conv(state)
#         val = self.fc_val(conv_out)
#         adv = self.fc_adv(conv_out)
#         return val + adv - adv.mean(dim=1, keepdim=True)

#     def get_q_values(self, state):
#         with torch.no_grad():
#             return self.forward(state)


# class WD3QNE:
#     def __init__(self,
#                  state_dim=37,
#                  num_actions=25,
#                  ensemble_size=3,
#                  gamma=0.99,
#                  tau=0.1):
#         self.device = device
#         self.ensemble_size = ensemble_size
#         self.num_actions = num_actions
#         self.gamma = gamma
#         self.tau = tau

#         # Ensemble Q-Networks
#         self.Q_ensemble = [WD3QNE_Net(state_dim, num_actions).to(device)
#                            for _ in range(ensemble_size)]
#         self.Q_target_ensemble = [copy.deepcopy(
#             q_net) for q_net in self.Q_ensemble]

#         self.optimizers = [torch.optim.Adam(q_net.parameters(), lr=0.0001)
#                            for q_net in self.Q_ensemble]

#     def train(self, batchs, epoch):
#         (state, next_state, action, next_action,
#          reward, done, bloc_num, SOFAS) = batchs
#         batch_s = 128
#         uids = np.unique(bloc_num)
#         num_batch = uids.shape[0] // batch_s
#         record_loss = []
#         sum_q_loss = 0
#         Batch = 0

#         for batch_idx in range(num_batch + 1):
#             batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
#             batch_user = np.isin(bloc_num, batch_uids)

#             # Ambil batch data
#             state_user = state[batch_user, :]
#             next_state_user = next_state[batch_user, :]
#             action_user = action[batch_user]
#             next_action_user = next_action[batch_user]
#             reward_user = reward[batch_user]
#             done_user = done[batch_user]
#             SOFAS_user = SOFAS[batch_user]

#             batch = (state_user, next_state_user, action_user,
#                      next_action_user, reward_user, done_user, SOFAS_user)

#             loss = self.compute_loss(batch)
#             sum_q_loss += loss
#             if Batch % 25 == 0:
#                 print('Epoch :', epoch, 'Batch :', Batch,
#                       'Average Loss :', sum_q_loss / (Batch + 1))
#                 record_loss1 = sum_q_loss / (Batch + 1)
#                 record_loss.append(record_loss1)

#             if Batch % 100 == 0:
#                 self.polyak_target_update()
#             Batch += 1

#         return record_loss

#     def compute_loss(self, batch):
#         state, next_state, action, next_action, reward, done, SOFA = batch
#         batch_size = state.shape[0]
#         end_multiplier = 1 - done
#         range_batch = torch.arange(batch_size).long().to(device)

#         total_loss = 0
#         for i, (q_net, q_target_net, optimizer) in enumerate(zip(self.Q_ensemble, self.Q_target_ensemble, self.optimizers)):
#             optimizer.zero_grad()

#             # Current Q-values
#             q_values = q_net(state)
#             q_value = q_values[range_batch, action]

#             # Next state Double Q-learning
#             with torch.no_grad():
#                 q_values_next = torch.stack(
#                     [net.get_q_values(next_state) for net in self.Q_ensemble]).mean(0)
#                 next_actions = q_values_next.argmax(dim=1)
#                 q_target_next = q_target_net(next_state)
#                 q_target_value = q_target_next[range_batch, next_actions]

#             target_q = reward + self.gamma * q_target_value * end_multiplier
#             loss = F.smooth_l1_loss(q_value, target_q)

#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         return total_loss / self.ensemble_size

#     # def polyak_target_update(self):
#     #     for q_net, q_target_net in zip(self.Q_ensemble, self.Q_target_ensemble):
#     #         for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
#     #             target_param.data.copy_(
#     #                 self.tau * param.data + (1 - self.tau) * target_param.data)

#     def polyak_target_update(self):
#         for q_net, q_target_net in zip(self.Q_ensemble, self.Q_target_ensemble):
#             for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
#                 target_param.data.lerp_(param.data, self.tau)

#     def get_action(self, state):
#         with torch.no_grad():
#             q_values = torch.stack([net.get_q_values(state)
#                                    for net in self.Q_ensemble]).mean(0)
#             return torch.argmax(q_values, dim=1)

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim
# import torch.nn.functional as F
# import copy

# # Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class WD3QNE_Net(nn.Module):
#     def __init__(self, state_dim, n_actions):
#         super(WD3QNE_Net, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#         )
#         self.fc_val = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#         self.fc_adv = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, n_actions)
#         )

#     def forward(self, state):
#         conv_out = self.conv(state)
#         val = self.fc_val(conv_out)
#         adv = self.fc_adv(conv_out)
#         return val + adv - adv.mean(dim=1, keepdim=True)

#     def get_q_values(self, state):
#         with torch.no_grad():
#             return self.forward(state)


# class WD3QNE:
#     def __init__(self, state_dim=37, num_actions=25, ensemble_size=3, gamma=0.99, tau=0.1):
#         self.device = device
#         self.ensemble_size = ensemble_size
#         self.num_actions = num_actions
#         self.gamma = gamma
#         self.tau = tau

#         # Ensemble Q-Networks
#         self.Q_ensemble = [WD3QNE_Net(state_dim, num_actions).to(
#             device) for _ in range(ensemble_size)]
#         self.Q_target_ensemble = [copy.deepcopy(
#             q_net) for q_net in self.Q_ensemble]

#         self.optimizers = [torch.optim.Adam(
#             q_net.parameters(), lr=0.0001) for q_net in self.Q_ensemble]

#     def train(self, batches, epoch):
#         (state, next_state, action, next_action,
#          reward, done, bloc_num, SOFAS) = batches

#         # Convert all batches to tensors on the appropriate device
#         state = torch.tensor(state).float().to(device)
#         next_state = torch.tensor(next_state).float().to(device)
#         action = torch.tensor(action).long().to(device)
#         next_action = torch.tensor(next_action).long().to(device)
#         reward = torch.tensor(reward).float().to(device)
#         done = torch.tensor(done).float().to(device)
#         bloc_num = torch.tensor(bloc_num).long().to(device)

#         batch_size = 128
#         uids = torch.unique(bloc_num)
#         num_batches = len(uids) // batch_size

#         record_loss = []
#         total_loss = 0

#         for batch_idx in range(num_batches + 1):
#             batch_uids = uids[batch_idx *
#                               batch_size: (batch_idx + 1) * batch_size]
#             batch_mask = torch.isin(bloc_num, batch_uids)

#             # Create batch tensors
#             batch = (
#                 state[batch_mask],
#                 next_state[batch_mask],
#                 action[batch_mask],
#                 next_action[batch_mask],
#                 reward[batch_mask],
#                 done[batch_mask],
#                 SOFAS[batch_mask],
#             )

#             loss = self.compute_loss(batch)
#             total_loss += loss

#             # Log progress every 25 batches
#             if batch_idx % 25 == 0:
#                 avg_loss = total_loss / (batch_idx + 1)
#                 print(
#                     f"Epoch: {epoch}, Batch: {batch_idx}, Average Loss: {avg_loss:.4f}")
#                 record_loss.append(avg_loss)

#             # Update target network every 100 batches
#             if batch_idx % 100 == 0:
#                 self.polyak_target_update()

#         return record_loss

#     def compute_loss(self, batch):
#         state, next_state, action, next_action, reward, done, SOFA = batch
#         end_multiplier = 1 - done
#         batch_size = state.size(0)
#         range_batch = torch.arange(batch_size, device=device)

#         total_loss = 0
#         for q_net, q_target_net, optimizer in zip(self.Q_ensemble, self.Q_target_ensemble, self.optimizers):
#             optimizer.zero_grad()

#             # Current Q-values
#             q_values = q_net(state)
#             q_value = q_values[range_batch, action]

#             # Next state Double Q-learning
#             with torch.no_grad():
#                 q_values_next = torch.stack(
#                     [net.get_q_values(next_state) for net in self.Q_ensemble]).mean(0)
#                 next_actions = q_values_next.argmax(dim=1)
#                 q_target_value = q_target_net(
#                     next_state)[range_batch, next_actions]

#             # Compute TD target
#             target_q = reward + self.gamma * q_target_value * end_multiplier
#             loss = F.smooth_l1_loss(q_value, target_q)

#             # Backpropagation and optimizer step
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         return total_loss / self.ensemble_size

#     def polyak_target_update(self):
#         with torch.no_grad():
#             for q_net, q_target_net in zip(self.Q_ensemble, self.Q_target_ensemble):
#                 for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
#                     target_param.data.mul_(
#                         1 - self.tau).add_(self.tau * param.data)

#     def get_action(self, state):
#         state = torch.tensor(state).float().to(device)
#         with torch.no_grad():
#             q_values = torch.stack([net.get_q_values(state)
#                                    for net in self.Q_ensemble]).mean(0)
#             return torch.argmax(q_values, dim=1).cpu().numpy()

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



# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim
# import torch.nn.functional as F
# import copy

# device = 'cpu'


# class WeightedDuelingDQN(nn.Module):
#     def __init__(self, state_dim, n_actions):
#         super(WeightedDuelingDQN, self).__init__()

#         # Shared Feature Layer
#         self.feature = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )

#         # Value Stream
#         self.fc_val = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#         # Advantage Stream
#         self.fc_adv = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_actions)
#         )

#         # Weight factor for advantage normalization
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight factor

#     def forward(self, state):
#         features = self.feature(state)
#         val = self.fc_val(features)  # Scalar Value Function
#         adv = self.fc_adv(features)  # Advantage Function

#         # Weighted advantage normalization
#         adv_weighted = adv - self.alpha * adv.mean(dim=1, keepdim=True)

#         return val + adv_weighted


# class WeightedDuelingDoubleDQN:
#     def __init__(self,
#                  state_dim=37,
#                  num_actions=25,
#                  device='cpu',
#                  gamma=0.999,
#                  tau=0.1):
#         self.device = device
#         self.Q = WeightedDuelingDQN(state_dim, num_actions).to(device)
#         self.Q_target = copy.deepcopy(self.Q)
#         self.tau = tau
#         self.gamma = gamma
#         self.num_actions = num_actions
#         self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)

#     def train(self, batchs, epoch):
#         (state, next_state, action, next_action,
#          reward, done, bloc_num, SOFAS) = batchs
#         batch_s = 128
#         uids = np.unique(bloc_num)
#         num_batch = uids.shape[0] // batch_s  # batch
#         record_loss = []
#         sum_q_loss = 0
#         Batch = 0

#         for batch_idx in range(num_batch + 1):
#             batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
#             batch_user = np.isin(bloc_num, batch_uids)
#             state_user = state[batch_user, :]
#             next_state_user = next_state[batch_user, :]
#             action_user = action[batch_user]
#             next_action_user = next_action[batch_user]
#             reward_user = reward[batch_user]
#             done_user = done[batch_user]
#             SOFAS_user = SOFAS[batch_user]

#             batch = (state_user, next_state_user, action_user,
#                      next_action_user, reward_user, done_user, SOFAS_user)
#             loss = self.compute_loss(batch)
#             sum_q_loss += loss.item()
#             self.optimizer.zero_grad()  # Gradient clearing
#             loss.backward()  # backward propagation
#             self.optimizer.step()  # update

#             if Batch % 25 == 0:
#                 print('Epoch :', epoch, 'Batch :', Batch,
#                       'Average Loss :', sum_q_loss / (Batch + 1))
#                 record_loss1 = sum_q_loss / (Batch + 1)
#                 record_loss.append(record_loss1)
#             if Batch % 100 == 0:
#                 self.polyak_target_update()
#             Batch += 1

#         return record_loss

#     def polyak_target_update(self):
#         for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
#             target_param.data.copy_(
#                 self.tau * param.data + (1 - self.tau) * target_param.data)

#     def compute_loss(self, batch):
#         state, next_state, action, next_action, reward, done, SOFA = batch
#         gamma = self.gamma
#         end_multiplier = 1 - done
#         batch_size = state.shape[0]
#         range_batch = torch.arange(batch_size).long().to(device)

#         # Q-value predictions
#         Q_values = self.Q(state)
#         Q_eval = Q_values[range_batch, action]

#         # Double DQN logic: use Q for action selection, Q_target for evaluation
#         with torch.no_grad():
#             Q_next = self.Q(next_state)
#             best_next_actions = torch.argmax(Q_next, dim=1)
#             Q_target_next = self.Q_target(next_state)
#             Q_target_eval = Q_target_next[range_batch, best_next_actions]
#             targetQ = reward + gamma * Q_target_eval * end_multiplier

#         # Loss computation
#         return nn.SmoothL1Loss()(targetQ, Q_eval)

#     def get_action(self, state):
#         with torch.no_grad():
#             Q_values = self.Q(state)
#             a_star = torch.argmax(Q_values, dim=1)
#             return a_star


device = 'cpu'


class WD3QNE_Net(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(WD3QNE_Net, self).__init__()

        # Shared feature extraction
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

        # Dueling architecture: value and advantage streams
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

        # Weighted advantage adjustment
        weight = 0.5  # Human embedding weight factor
        adv_weighted = adv - adv.mean(dim=1, keepdim=True)
        weighted_q = val + (1 + weight) * adv_weighted
        return weighted_q


class WD3QNE(object):
    def __init__(self,
                 state_dim=37,
                 num_actions=25,
                 gamma=0.999,
                 tau=0.1):
        self.device = device
        self.Q = WD3QNE_Net(state_dim, num_actions).to(device)
        # self.Q = nn.DataParallel(self.Q)
        self.Q_target = copy.deepcopy(self.Q)
        self.tau = tau
        self.gamma = gamma
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)

    def train(self, batchs, epoch):
        (state, next_state, action, next_action,
         reward, done, bloc_num, SOFAS) = batchs
        batch_s = 128
        uids = np.unique(bloc_num)
        num_batch = uids.shape[0] // batch_s  # batch
        record_loss = []
        sum_q_loss = 0
        Batch = 0

        for batch_idx in range(num_batch + 1):
            batch_uids = uids[batch_idx * batch_s: (batch_idx + 1) * batch_s]
            batch_user = np.isin(bloc_num, batch_uids)
            state_user = state[batch_user, :]
            next_state_user = next_state[batch_user, :]
            action_user = action[batch_user]
            next_action_user = next_action[batch_user]
            reward_user = reward[batch_user]
            done_user = done[batch_user]
            SOFAS_user = SOFAS[batch_user]
            batch = (state_user, next_state_user, action_user,
                     next_action_user, reward_user, done_user, SOFAS_user)

            loss = self.compute_loss(batch)
            sum_q_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if Batch % 25 == 0:
                print('Epoch :', epoch, 'Batch :', Batch,
                      'Average Loss :', sum_q_loss / (Batch + 1))
                record_loss.append(sum_q_loss / (Batch + 1))

            if Batch % 100 == 0:
                self.polyak_target_update()
            Batch += 1

        return record_loss

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def compute_loss(self, batch):
        state, next_state, action, next_action, reward, done, SOFA = batch
        gamma = 0.99
        end_multiplier = 1 - done
        batch_size = state.shape[0]
        range_batch = torch.arange(batch_size).long().to(device)

        # Q-value estimation
        Q_values = self.Q(state)
        Q_values_action = Q_values[range_batch, action]

        with torch.no_grad():
            # Double DQN Target computation
            Q_next_values = self.Q(next_state)
            Q_next_actions = torch.argmax(Q_next_values, dim=1)
            Q_next_target = self.Q_target(next_state)
            Q_next = Q_next_target[range_batch, Q_next_actions]

        # Weighted embedding
        human_weight = 0.3  # Adjustable human factor
        reward_weighted = reward * (1 + human_weight)
        target_Q = reward_weighted + (gamma * Q_next * end_multiplier)

        return nn.SmoothL1Loss()(Q_values_action, target_Q)

    def get_action(self, state):
        with torch.no_grad():
            Q_values = self.Q(state)
            action = torch.argmax(Q_values, dim=1)
            return action
