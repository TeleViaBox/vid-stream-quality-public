# subtask1
```
具体整合步骤:

1. 数据整合
从LeoEM的第一阶段获取卫星动态位置数据，结合车辆的实时位置数据输入V2V和V2I通道模型。
在每个时间步，根据车辆和卫星的位置，计算路径损耗和阴影衰落，并更新通道模型。

2. 路径计算
使用LeoEM第二阶段的预计算路径，结合Dijkstra算法计算车辆与卫星之间的最短路径。
在路径计算中考虑链路延迟和干扰，优化传输路径。

3. 实时仿真
在LeoEM第三阶段的仿真环境中，集成V2V和V2I通道模型。
使用SaTCP进行TCP适配，通过减少切换时的传输中断，优化网络性能。
在仿真过程中实时监控传输质量，调整路径和传输参数。
```

# related works and references
1. MARLspectrumSharingV2X: https://github.com/le-liang/MARLspectrumSharingV2X
2. LeoEM: https://github.com/XuyangCaoUCSD/LeoEM

```
Why important for my repo:
- Important concepts inside "LeoEM": this repo generate data of simulated LEO behavior.
```

# file structure
```
LEO_to_Vehicle_Network/
│
├── data/
│   ├── satellite_positions/           # 卫星位置数据
│   ├── vehicle_positions/             # 车辆位置数据
│   ├── ground_stations.xlsx           # 地面站位置数据
│   ├── precomputed_paths/             # 预计算路径数据
│   └── environmental_conditions/      # 环境条件数据，如天气等
│
├── models/
│   ├── path_loss.py                   # 路径损耗模型
│   ├── shadowing.py                   # 阴影衰落模型
│   ├── channel_model.py               # 通道模型更新
│   ├── multipath_effects.py           # 多径效应模型
│   ├── doppler_effect.py              # 多普勒效应模型
│   ├── antenna_gain.py                # 天线增益模型
│   ├── interference.py                # 干扰模型
│   ├── atmospheric_attenuation.py     # 大气衰减模型
│   ├── rain_attenuation.py            # 雨衰模型
│   ├── handover_effects.py            # 切换效应模型
│   └── link_budget.py                 # 链路预算计算
│
├── rl_optimization/
│   ├── rl_agent.py                    # 强化学习代理
│   ├── environment.py                 # 强化学习环境
│   ├── train.py                       # 训练脚本
│   └── test.py                        # 测试脚本
│
├── utils/
│   ├── data_loader.py                 # 数据加载工具
│   ├── metrics.py                     # 性能指标计算工具
│   └── visualization.py               # 可视化工具
│
├── scripts/
│   ├── preprocess_data.py             # 数据预处理脚本
│   ├── generate_paths.py              # 路径生成脚本
│   ├── run_simulation.py              # 运行仿真脚本
│   └── evaluate_performance.py        # 性能评估脚本
│
├── results/                           # 结果目录
│   ├── figures/                       # 图表
│   ├── logs/                          # 日志
│   └── reports/                       # 报告
│
├── main.py                            # 主程序入口
└── README.md                          # 项目说明文件

```
