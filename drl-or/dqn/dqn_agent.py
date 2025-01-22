import os
import time

import torch
from torch import nn
import torch.nn.functional as F
import random

import numpy as np

# two hidden layer mlp
class MLPQNet(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = self.out(x)
        return action_prob

# Dueling DQN Network
# A = Q - V
# 利用优势函数和价值函数估计Q函数
class VAnet(nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

# 经验回放内存占用问题
# 使用优先级经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # experience: state, action, reward, next_state, (done)
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, state_size, action_size, hidden_size=64,
                 lr=1e-3, epsilon=0.2, epsilon_min=0.005, epsilon_decay=0.999, target_update_freq=10, gamma=0.9,
                 memory_capacity=50_000, batch_size=32, replay_start_size=1000, save_model_step=10_000,
                 dqn_type= "VanillaDQN",
                 nn_base = "MLP",
                 is_train=True,
                 use_double_dqn=False):

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if dqn_type == "VanillaDQN":

            if nn_base == "MLP":
                # 评估网络（q网络）
                self.eval_net = MLPQNet(state_size, action_size, hidden_size)
                # 目标网络
                self.target_net = MLPQNet(state_size, action_size, hidden_size)

        # 竞争DQN
        elif dqn_type == "DuelingDQN":
            self.eval_net = VAnet(state_size, action_size, hidden_size)
            self.target_net = VAnet(state_size, action_size, hidden_size)


        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        self.learn_step = 0

        # Adam参数设置， 学习率衰减
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        # 目标网络更新频率
        self.target_update_freq = target_update_freq
        # epsilon-greedy 策略
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 衰减因子
        # 折扣因子 0.9
        self.gamma = gamma

        self.replay_memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        
        self.save_model_step = save_model_step

        # 指定DQN type
        self.dqn_type = dqn_type
        self.use_double_dqn = use_double_dqn
        self.is_train = is_train

    # 这里的随机探索策略可以随着训练增加降低探索率
    # 训练模式，遵循ε-greedy策略，根据epsilon随机选择动作
    # 测试模式，直接选择q值最大的动作
    def choose_action(self, state) -> int:
        # ε-greedy 策略选择动作
        if self.is_train and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)

        else:
            # if state is already a tensor
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            # action = self.eval_net(state).argmax().item()
            action = self.eval_net(state).argmax().item()

        return action

    def predict_batch_q_values(self, node_feat, edge_index, edge_attr, batch_node_feat_action, batch_edge_attr_action):
        if self.is_train and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            q_value_list = []
            for i in range(len(batch_node_feat_action)):
                node_feat_action = batch_node_feat_action[i]
                edge_attr_action = batch_edge_attr_action[i]
                q_value = self.eval_net(node_feat_action, edge_index, edge_attr_action)
                q_value_list.append(q_value)
            action = np.argmax(q_value_list)
        return action


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self):

        if len(self.replay_memory) < self.replay_start_size:
            return

        if self.learn_step % self.target_update_freq == 0:
            # 渐进更新目标网络参数
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        batch_data = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states= zip(*batch_data)

        # 这里state输入：tensor或者numpy, 如果是numpy，需要转为tensor
        # 堆叠每个状态的
        states = torch.stack(states, dim=0).to(self.device)
        # states = torch.tensor(torch.hstack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        rewards= torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.stack(next_states, dim=0).to(self.device)
        # next_states = torch.tensor(torch.hstack(next_states), dtype=torch.float32).to(self.device)

        q_next = self.target_net(next_states).max(1)[0].view(-1, 1)
        if self.use_double_dqn:
            # 选择q网络下最大的动作，再计算target_net网络下的值,避免过高估计Q值
            max_action = self.eval_net(next_states).max(1)[1].view(-1, 1)
            q_next = self.target_net(next_states).gather(1, max_action)
        q_value = self.eval_net(states).gather(1, actions)
        q_target = rewards + self.gamma * q_next
        dqn_loss = torch.mean(F.mse_loss(q_value , q_target))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        self.update_epsilon()

        if self.learn_step % self.save_model_step == 0:
            self.save_model()

    def gnn_learn(self):
        pass


    def store_transition(self, state, action, reward, next_state):
        self.replay_memory.push((state, action, reward, next_state))

    def save_model(self, save_dir='./dqn_checkpoints'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # 构建文件名（根据时间戳）
        model_filename = os.path.join(save_dir, f"dqn_model_step_{self.learn_step}_{timestamp}.pth")
        # 保存DQN模型的权重
        torch.save(self.eval_net.state_dict(), model_filename)
        print(f"DQN模型权重已保存到 {model_filename}")


if __name__ == "__main__":

    dqn_agent = DQNAgent(4, 4)
    import numpy as np
    for _ in range(100):
        state = torch.randn(4)  # 4维状态向量示例
        action = random.randint(0, 1)  # 假设是二选一动作
        reward = random.random()  # 随机奖励
        next_state = torch.randn(4)  # 下一个状态
        dqn_agent.choose_action(state)
        dqn_agent.replay_memory.push((state, action, reward, next_state))

    dqn_agent.learn()






