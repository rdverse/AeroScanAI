import torch
from torch.utils.data import Dataset
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# IMPORTANT : IF YOU NEED TO MODIFY THE DEFECT GO TO CHECKPOINT101 ( in comments )


class simulatedDataset(Dataset):
    def __init__(self, img_dim, n_channels, n_samples, defect_coverage=0.75):
        self.img_dim = img_dim
        self.n_channels = n_channels
        self.defect_coverage = defect_coverage
        self.random_seeds = np.arange(0, n_samples, n_samples)
        self.scale = MinMaxScaler()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, mask = self.synthetic_defects(self.random_seeds[idx])
        data = np.round(self.scale.fit_transform(self.data.reshape(-1, 10)).reshape(self.img_dim, self.img_dim, self.n_channels), 3)
        
        sample = {
            'data': torch.tensor(data, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        }
        return sample

    def synthetic_defects(self, random_seed=0):
        np.random.seed(random_seed)
        def random_sine_wave(length, defect=True):
            # Generate a random amplitude
            amplitude = random.uniform(0.1, 1.0)
            # Generate a random frequency
            frequency = random.uniform(1, 20)
            frequency = np.pi * 0.5
            # Generate a random phase
            phase = random.uniform(0, np.pi * 0.1)
            # Generate the sine wave
            sine_wave = 1 * np.sin(2 * np.pi * frequency * np.arange(length) + phase) +1
            # Add noise
            noise_level = 1
            noise = np.random.random_sample(noise_level)
            
            # CHECKPOINT101 : DEFECTS ARE ADDED ADDITIONAL NOISE. YOU CAN CHOOSE TO REMOVE
            if defect:
                noise = noise + np.random.randint(0, 3, 10)
            sine_wave += noise
            return sine_wave

        data = np.zeros((self.img_dim, self.img_dim, 10))
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
                        data[i][j] = random_sine_wave(10, defect=True)
                    else:
                        data[i][j] = random_sine_wave(10, defect=False)

        return data, mask, pixelmap
