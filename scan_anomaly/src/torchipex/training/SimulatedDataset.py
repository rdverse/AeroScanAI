import torch
from torch.utils.data import Dataset
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# IMPORTANT : IF YOU NEED TO MODIFY THE DEFECT GO TO CHECKPOINT101 ( in comments )
import math
class SimulatedDataset(Dataset):
    def __init__(self, img_dim, 
                 n_channels, 
                 n_samples, 
                 defect_coverage=0.75, 
                 random_seed=0):
        self.img_dim = img_dim
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.defect_coverage = defect_coverage
        self.random_seed = random_seed*10000
        self.random_seeds = np.arange(self.random_seed, 
                                      self.random_seed+self.n_samples, 
                                      1)
        self.scale = MinMaxScaler()
        self.random_seed = random_seed
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data, mask, pixelmap = self.synthetic_defects(self.random_seeds[idx])
        data = self.scale.fit_transform(data.reshape(-1, self.n_channels)).reshape(self.img_dim, self.img_dim, self.n_channels)
        # rounding does not reduce memory usage!
         
        sample = {
            'data': torch.tensor(data, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        }
        return sample

    def synthetic_defects(self, current_seed):
        np.random.seed(current_seed)
        def random_sine_wave(length, defect=True):
            np.random.seed(current_seed)
            # Generate a random amplitude
            amplitude = random.uniform(0.1, 1.0)
            # Generate a random frequency
            frequency = np.pi * 0.5
            # Generate a random phase
            phase = random.uniform(0, np.pi * 0.1)
            # Generate the sine wave
            sine_wave = 1 * np.sin(2 * np.pi * frequency * np.arange(length) + phase) +1
            # Add noise
            noise_level = 1
            noise =  np.random.uniform(0, noise_level, self.n_channels)
            
            # CHECKPOINT101 : DEFECTS ARE ADDED ADDITIONAL NOISE. YOU CAN CHOOSE TO REMOVE
            if defect:
                noise = noise + np.random.uniform(0, 1, self.n_channels)
                dampen_width = math.ceil(self.n_channels//10)
                n_dampen_points = np.random.choice([1,2,3])
                if dampen_width>0:
                    for i in range(n_dampen_points):
                        dampen_start = np.random.randint(0, self.n_channels - dampen_width - 1)
                        dampen_end = dampen_start + dampen_width
                        sine_wave[dampen_start:dampen_end] = 0
            sine_wave += noise
            return sine_wave
        np.random.seed(current_seed)
        data = np.zeros((self.img_dim, self.img_dim, self.n_channels))
        mask = np.zeros((self.img_dim, self.img_dim))
        pixelmap = np.zeros((self.img_dim, self.img_dim, 2)) # 2 for x and y
        defect_area = int(self.img_dim * self.defect_coverage)

        for i in range(defect_area):
            num_points = defect_area
            points = []
            for j in range(num_points):
                x = random.randint(0, self.img_dim - 1)
                y = random.randint(0, self.img_dim - 1)
                points.append((x, y))

            for j in range(num_points - 1):
                x1, y1 = points[j][1]-1, points[j][0]-1
                x2, y2 = points[j + 1][1]-1, points[j + 1][0]-1
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                x2 = 0 if x2 < 0 else x2
                y2 = 0 if y2 < 0 else y2
                mask[x1][y1] = 1
                mask[x2][y2] = 1

        for i in range(self.img_dim):
            for j in range(self.img_dim):
                if np.sum(data[i][j]) > 0:
                    continue
                else:
                    if mask[i][j] == 1:
                        data[i][j] = random_sine_wave(self.n_channels, defect=True)
                    else:
                        data[i][j] = random_sine_wave(self.n_channels, defect=False)

        return data, mask, pixelmap