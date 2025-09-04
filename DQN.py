import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import time
import platform  # 用于检测操作系统

# 设置随机种子，保证结果可复现
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 经验回放缓冲区
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """存储经验"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """随机采样一批经验"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前存储的经验数量"""
        return len(self.memory)


# DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, lr=5e-4,
                 epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_capacity=50000,
                 target_update=10,
                 is_double_dqn=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.lr = lr  # 学习率
        self.epsilon = epsilon_start  # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update  # 目标网络更新频率
        self.is_double_dqn = is_double_dqn  # 是否使用Double DQN

        # 策略网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练

        # 优化器和经验回放缓冲区
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)

        # 记录训练步数
        self.steps_done = 0

    def select_action(self, state):
        """根据当前状态选择动作（epsilon-greedy策略）"""
        self.steps_done += 1
        # 随机选择动作（探索）
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        # 选择最优动作（利用）
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def optimize_model(self):
        """从经验回放中学习，更新策略网络"""
        if len(self.memory) < self.batch_size:
            return  # 经验不足时不更新

        # 采样一批经验
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # 转换为张量
        state_batch = torch.tensor(batch.state, dtype=torch.float32)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, dtype=torch.float32)

        # 计算当前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        if self.is_double_dqn:
            # Double DQN: 用策略网络选择动作，目标网络评估价值
            with torch.no_grad():
                next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                next_q = self.target_net(next_state_batch).gather(1, next_actions)
        else:
            # 普通DQN: 直接用目标网络选择并评估
            with torch.no_grad():
                next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)

        target_q = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * next_q

        # 计算损失并优化
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练函数
def train_agent(env, agent, agent_name, num_episodes=500, max_steps=500):
    scores = []  # 记录每回合的奖励
    losses = []  # 记录损失
    solved = False  # 标记是否解决问题

    print(f"\n===== 开始训练 {agent_name} =====")
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        score = 0
        episode_loss = 0
        steps = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 更新状态和分数
            state = next_state
            score += reward
            steps += 1

            # 优化模型
            loss = agent.optimize_model()
            if loss is not None:
                episode_loss += loss

            # 更新目标网络
            agent.update_target_network()

            if done:
                break

        # 更新探索率
        agent.update_epsilon()

        # 记录结果
        scores.append(score)
        avg_loss = episode_loss / steps if steps > 0 else 0
        losses.append(avg_loss)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_score_10 = np.mean(scores[-10:])
            avg_score_100 = np.mean(scores[-100:]) if len(scores) >= 100 else 0
            print(f"Episode {episode + 1}/{num_episodes}, 得分: {score:.2f}, "
                  f"近10集平均: {avg_score_10:.2f}, 近100集平均: {avg_score_100:.2f}, "
                  f"探索率: {agent.epsilon:.3f}, 平均损失: {avg_loss:.6f}")

        # 检查是否解决问题（连续100回合平均奖励>=195）
        if not solved and len(scores) >= 100:
            if np.mean(scores[-100:]) >= 195:
                print(f"\n{agent_name} 问题解决！在第 {episode + 1} 回合达到目标。")
                solved = True
                # 提前停止训练
                break

    return scores, losses, solved


# 测试函数
def test_agent(env, agent, agent_name, num_episodes=5, render=False):
    total_rewards = []
    print(f"\n===== 测试 {agent_name} =====")
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        score = 0
        for _ in range(500):  # 最大步数
            if render:
                env.render()
                time.sleep(0.01)  # 放慢速度以便观察

            # 选择最优动作（不探索）
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = agent.policy_net(state_tensor).max(1)[1].item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            score += reward

            if done:
                break

        total_rewards.append(score)
        print(f"测试回合 {episode + 1}, 得分: {score}")

    avg_score = np.mean(total_rewards)
    print(f"{agent_name} 平均测试分数: {avg_score:.2f}")
    return avg_score


# 绘制训练曲线 - 已修复中文显示问题
def plot_results(scores_dqn, scores_ddqn, losses_dqn, losses_ddqn, solved_dqn, solved_ddqn):
    # 解决中文显示问题 - 根据操作系统自动选择字体
    system = platform.system()
    if system == 'Windows':
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    elif system == 'Darwin':  # macOS
        plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS"]
    else:  # Linux
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 计算滑动平均以平滑曲线
    def moving_average(values, window=10):
        return [np.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制奖励曲线
    ax1.plot(moving_average(scores_dqn), label='基准DQN')
    ax1.plot(moving_average(scores_ddqn), label='Double DQN')
    ax1.axhline(y=195, color='r', linestyle='--', label='目标分数')

    # 标记解决问题的时刻
    if solved_dqn:
        ax1.axvline(x=len(scores_dqn) - 1, color='blue', linestyle=':', label=f'基准DQN解决于{len(scores_dqn)}回合')
    if solved_ddqn:
        ax1.axvline(x=len(scores_ddqn) - 1, color='green', linestyle=':',
                    label=f'Double DQN解决于{len(scores_ddqn)}回合')

    ax1.set_title('每回合奖励（滑动平均）')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.legend()
    ax1.grid(True)

    # 绘制损失曲线
    ax2.plot(moving_average(losses_dqn), label='基准DQN')
    ax2.plot(moving_average(losses_ddqn), label='Double DQN')
    ax2.set_title('每回合损失（滑动平均）')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 定义基准DQN超参数（针对其不稳定性进行调整）
    dqn_params = {
        'gamma': 0.99,
        'lr': 3e-4,  # 更低的学习率，提高稳定性
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,  # 更慢的探索率衰减
        'batch_size': 64,
        'buffer_capacity': 100000,  # 更大的经验池
        'target_update': 20,  # 目标网络更新频率更低
        'is_double_dqn': False
    }

    # 定义Double DQN超参数
    ddqn_params = {
        'gamma': 0.99,
        'lr': 5e-4,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99,  # 可以更快衰减
        'batch_size': 64,
        'buffer_capacity': 50000,
        'target_update': 10,
        'is_double_dqn': True
    }

    # 训练基准DQN
    dqn_agent = DQNAgent(state_dim, action_dim, **dqn_params)
    scores_dqn, losses_dqn, solved_dqn = train_agent(
        env, dqn_agent, "基准DQN", num_episodes=500
    )

    # 训练Double DQN
    ddqn_agent = DQNAgent(state_dim, action_dim, **ddqn_params)
    scores_ddqn, losses_ddqn, solved_ddqn = train_agent(
        env, ddqn_agent, "Double DQN", num_episodes=500
    )

    # 测试算法
    env_test = gym.make('CartPole-v1', render_mode='human')
    dqn_avg_score = test_agent(env_test, dqn_agent, "基准DQN", render=True)
    ddqn_avg_score = test_agent(env_test, ddqn_agent, "Double DQN", render=True)
    env_test.close()

    # 绘制结果
    plot_results(scores_dqn, scores_ddqn, losses_dqn, losses_ddqn, solved_dqn, solved_ddqn)

    # 打印最终比较结果
    print("\n===== 算法比较 =====")
    print(f"基准DQN 解决状态: {'已解决' if solved_dqn else '未解决'}")
    print(f"Double DQN 解决状态: {'已解决' if solved_ddqn else '未解决'}")
    print(f"基准DQN 平均测试分数: {dqn_avg_score:.2f}")
    print(f"Double DQN 平均测试分数: {ddqn_avg_score:.2f}")

    env.close()


if __name__ == "__main__":
    main()
