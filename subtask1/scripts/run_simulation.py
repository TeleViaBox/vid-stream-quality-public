import numpy as np
from rl_optimization.leo_vehicle_env import LeoVehicleEnv
from utils.data_loader import load_data

def run_simulation():
    # 加载数据
    satellite_positions = load_data('data/satellite_positions')
    vehicle_positions = load_data('data/vehicle_positions')

    # 初始化环境
    env = LeoVehicleEnv(satellite_positions, vehicle_positions, other_params={})

    # 运行仿真
    state = env.reset()
    done = False
    while not done:
        action = np.random.choice(env.action_space.n)  # 随机选择动作，实际应用中应使用训练好的代理
        next_state, reward, done, _ = env.step(action)
        state = next_state

    # 保存仿真结果
    simulation_results = {
        'final_state': state
    }
    np.save('results/logs/simulation_results.npy', simulation_results)

if __name__ == "__main__":
    run_simulation()
