import numpy as np
from rl_optimization.rl_agent import RLAgent
from rl_optimization.leo_vehicle_env import LeoVehicleEnv
from utils.data_loader import load_data
from utils.visualization import plot_results

def train():
    # 加载数据
    satellite_positions = load_data('data/satellite_positions')
    vehicle_positions = load_data('data/vehicle_positions')

    # 初始化环境和代理
    env = LeoVehicleEnv(satellite_positions, vehicle_positions, other_params={})
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = RLAgent(state_size, action_size)

    # 训练参数
    episodes = 1000

    # 训练过程
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.train(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}")
                break

    # 保存结果
    results = {
        'episodes': episodes,
        'agent': agent
    }
    np.save('results/logs/training_results.npy', results)

    # 可视化结果
    plot_results(results)

if __name__ == "__main__":
    train()
