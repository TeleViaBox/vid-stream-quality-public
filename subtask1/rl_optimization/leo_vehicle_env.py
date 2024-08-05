import numpy as np
import gym
from gym import spaces
from models.path_loss import calculate_path_loss
from models.shadowing import calculate_shadowing
# from models.doppler_effect import calculate_doppler_shift

class LeoVehicleEnv(gym.Env):
    def __init__(self, satellite_positions, vehicle_positions, other_params):
        super(LeoVehicleEnv, self).__init__()
        self.satellite_positions = satellite_positions
        self.vehicle_positions = vehicle_positions
        self.other_params = other_params
        self.action_space = spaces.Discrete(len(satellite_positions))  # 选择卫星的动作空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,))  # 示例状态空间

    def reset(self):
        # 重置环境状态
        self.current_step = 0
        self.state = self._get_state()
        return self.state

    def step(self, action):
        # 执行动作，更新环境状态
        reward, done = self._take_action(action)
        self.current_step += 1
        self.state = self._get_state()
        return self.state, reward, done, {}

    def _get_state(self):
        # 获取当前状态
        vehicle_pos = self.vehicle_positions[self.current_step]
        satellite_pos = self.satellite_positions
        path_loss = calculate_path_loss(vehicle_pos, satellite_pos)
        shadowing = calculate_shadowing()
        doppler_shift = calculate_doppler_shift(vehicle_pos, satellite_pos)
        state = np.array([path_loss, shadowing, doppler_shift])
        return state

    def _take_action(self, action):
        # 根据动作计算奖励和是否结束
        selected_satellite = self.satellite_positions[action]
        vehicle_pos = self.vehicle_positions[self.current_step]
        path_loss = calculate_path_loss(vehicle_pos, selected_satellite)
        latency = path_loss  # 这里简化为延迟与路径损耗成正比
        reward = -latency  # 延迟越小，奖励越大
        done = self.current_step >= len(self.vehicle_positions) - 1
        return reward, done

    def render(self, mode='human'):
        # 可视化
        pass
