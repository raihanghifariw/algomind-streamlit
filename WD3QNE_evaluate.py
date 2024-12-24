import torch
import numpy as np
import torch.nn.functional as F

# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def do_eval(model, batchs):
    """
    Evaluasi model WD3QNE pada satu batch data.
    Mengembalikan Q-value, aksi agen, aksi fisik, dan probabilitas aksi.
    """
    state, next_state, action, next_action, reward, done = batchs
    # Ambil Q-value dari semua jaringan ensembel dan rata-rata
    with torch.no_grad():
        Q_values_ensemble = torch.stack(
            [net.get_q_values(state) for net in model.Q_ensemble])
        Q_values_mean = Q_values_ensemble.mean(
            0)  # Rata-rata Q dari semua jaringan

    # Aksi agen dari Q-value rata-rata
    agent_actions = torch.argmax(Q_values_mean, dim=1)
    phy_actions = action

    # Hitung probabilitas aksi menggunakan softmax
    Q_value_pro1 = F.softmax(Q_values_mean, dim=1)
    Q_value_pro_ind = torch.argmax(Q_value_pro1, dim=1)
    Q_value_pro = Q_value_pro1[torch.arange(
        len(Q_value_pro_ind)), Q_value_pro_ind]

    return Q_values_mean, agent_actions, phy_actions, Q_value_pro


# def do_eval(model, batchs):
#     """
#     Evaluasi model WD3QNE pada satu batch data.
#     Mengembalikan Q-value, aksi agen, aksi fisik, dan probabilitas aksi.
#     """
#     state, next_state, action, next_action, reward, done = batchs
#     with torch.no_grad():
#         # Hitung Q-value menggunakan jaringan utama Q
#         Q_values = model.Q(state)

#     # Aksi agen berdasarkan Q-values
#     agent_actions = torch.argmax(Q_values, dim=1)
#     phys_actions = action  # Aksi fisik dari batch

#     # Hitung probabilitas aksi dengan softmax
#     Q_value_pro1 = F.softmax(Q_values, dim=1)
#     Q_value_pro = Q_value_pro1[torch.arange(len(agent_actions)), agent_actions]

#     return Q_values, agent_actions, phys_actions, Q_value_pro



def do_test(model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    """
    Evaluasi model WD3QNE pada dataset uji.
    Menghasilkan dan menyimpan hasil evaluasi dalam file numpy.
    """
    bloc_max = max(bloctest)
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]

    RNNstate = Xtest
    print('####  Generating test set traces  ####')
    statesize = RNNstate.shape[1]
    num_samples = RNNstate.shape[0]

    states = np.zeros((num_samples, statesize))
    actions = np.zeros((num_samples, 1), dtype=int)
    next_actions = np.zeros((num_samples, 1), dtype=int)
    rewards = np.zeros((num_samples, 1))
    next_states = np.zeros((num_samples, statesize))
    done_flags = np.zeros((num_samples, 1))
    bloc_num = np.zeros((num_samples, 1))

    c = 0
    blocnum1 = 1

    for i in range(num_samples - 1):
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1

        if bloctest[i + 1] == 1:  # Akhir episode pasien
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 += 1
            reward1 = (-beat[0] * (SOFA[i]) + R3[i])
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            reward1 = (-beat[1] * (SOFA[i + 1] - SOFA[i]))

        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c += 1

    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1

    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    reward1 = -beat[0] * (SOFA[c]) + R3[c]

    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c += 1

    bloc_num = np.squeeze(bloc_num[:c, :])
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = np.squeeze(actions[: c, :])
    next_actions = np.squeeze(next_actions[: c, :])
    rewards = np.squeeze(rewards[: c, :])
    done_flags = np.squeeze(done_flags[: c, :])

    # Tensor conversion
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.LongTensor(actions).to(device)
    next_action = torch.LongTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)
    batchs = (state, next_state, action, next_action, reward, done, bloc_num)

    # Rekaman hasil evaluasi
    rec_phys_q, rec_agent_q, rec_agent_q_pro = [], [], []
    rec_phys_a, rec_agent_a, rec_sur, rec_reward_user = [], [], [], []

    batch_size = 128
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_size

    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_mask = np.isin(bloc_num, batch_uids)
        batch = (state[batch_mask], next_state[batch_mask], action[batch_mask],
                 next_action[batch_mask], reward[batch_mask], done[batch_mask])

        q_output, agent_actions, phys_actions, Q_value_pro = do_eval(
            model, batch)
        q_output_len = torch.arange(len(q_output))
        agent_q = q_output[q_output_len, agent_actions]
        phys_q = q_output[q_output_len, phys_actions]

        rec_agent_q.extend(agent_q.cpu().numpy())
        rec_agent_q_pro.extend(Q_value_pro.cpu().numpy())
        rec_phys_q.extend(phys_q.cpu().numpy())
        rec_agent_a.extend(agent_actions.cpu().numpy())
        rec_phys_a.extend(phys_actions.cpu().numpy())
        rec_sur.extend(Y90[batch_mask])
        rec_reward_user.extend(reward[batch_mask].cpu().numpy())

    # Simpan hasil evaluasi
    np.save('WD3QNE-algorithm/shencunlv.npy', rec_sur)
    np.save('WD3QNE-algorithm/agent_bQ.npy', rec_agent_q)
    np.save('WD3QNE-algorithm/phys_bQ.npy', rec_phys_q)
    np.save('WD3QNE-algorithm/reward.npy', rec_reward_user)
    np.save('WD3QNE-algorithm/agent_actionsb.npy', rec_agent_a)
    np.save('WD3QNE-algorithm/phys_actionsb.npy', rec_phys_a)
    np.save('WD3QNE-algorithm/rec_agent_q_pro.npy', rec_agent_q_pro)
