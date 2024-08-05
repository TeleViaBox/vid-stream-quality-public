# leos-vehicle-network

# main task (To-dos)
1. data source
2. consider mutliple effects in LEOs and vehicular network
3. plots of results (see followings)
4. Use this repo as a sub-module, for this new repo: implemntation on OMNeT++, and NS-3
5. license writing (giving credits to my reference repos) 

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

# current (not goal) file structrue
```
.
├── README.md
├── data
│   ├── environmental_conditions
│   ├── ground_stations.xlsx
│   ├── precomputed_paths
│   ├── satellite_positions
│   └── vehicle_positions
├── main.py
├── models
│   ├── __pycache__
│   │   ├── path_loss.cpython-312.pyc
│   │   └── shadowing.cpython-312.pyc
│   ├── channel_model.py
│   ├── path_loss.py
│   └── shadowing.py
├── requirements.txt
├── rl_optimization
│   ├── __pycache__
│   │   ├── leo_vehicle_env.cpython-312.pyc
│   │   ├── rl_agent.cpython-312.pyc
│   │   └── train.cpython-312.pyc
│   ├── environment.py
│   ├── leo_vehicle_env.py
│   ├── rl_agent.py
│   ├── test.py
│   └── train.py
├── scripts
│   ├── generate_paths.py
│   └── run_simulation.py
└── utils
    ├── data_loader.py
    ├── metrics.py
    └── visualization.py

11 directories, 22 files
```


# plots of results
```
路径损耗随时间的变化
横轴：时间（秒）
纵轴：路径损耗（dB）
效果：展示不同时间点车辆与卫星之间路径损耗的变化，反映出动态路径调整后的损耗情况。

阴影衰落随时间的变化
横轴：时间（秒）
纵轴：阴影衰落（dB）
效果：展示不同时间点阴影衰落的变化情况，反映出动态环境下阴影衰落对信道的影响。

传输速率随时间的变化
横轴：时间（秒）
纵轴：传输速率（Mbps）
效果：展示不同时间点的传输速率，反映出路径优化后传输性能的变化。

传输延迟随时间的变化
横轴：时间（秒）
纵轴：传输延迟（毫秒）
效果：展示不同时间点的传输延迟，反映出路径优化后延迟的改进情况。

TCP吞吐量随时间的变化
横轴：时间（秒）
纵轴：TCP吞吐量（Mbps）
效果：展示不同时间点的TCP吞吐量，反映出使用SaTCP优化后的TCP性能变化。
```


# error fixed

##### error-01

This is caused by: requiremnets.txt
```
  File "/tmp/pip-build-env-j_bxv_il/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 18, in <module>
    from setuptools.extern.six import PY3, string_types
ModuleNotFoundError: No module named 'setuptools.extern.six'

(base) xxx@xxx:~/xxx/leos-vehicle-network/subtask1$ 
```