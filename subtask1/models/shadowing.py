import numpy as np

def calculate_shadowing(std_dev=3, size=1):
    return np.random.normal(0, std_dev, size)
