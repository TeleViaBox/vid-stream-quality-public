import numpy as np

def calculate_path_loss(vehicle, satellite):
    distance = np.linalg.norm(np.array(vehicle) - np.array(satellite))
    return 128.1 + 37.6 * np.log10(distance)
