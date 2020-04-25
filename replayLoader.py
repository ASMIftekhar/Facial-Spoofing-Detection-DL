import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import imageio


class replayLoader(Dataset):

    def __init__(self, csv_file, root_dir, frames_ps):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
        """
        reader = pd.read_csv(csv_file)
        self.samples = reader
        self.root_dir = root_dir
        self.frames_ps = frames_ps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.samples.iloc[idx, 0])
        video = imageio.get_reader(img_name, "ffmpeg")

        jump = 25/self.frames_ps # Assuming all videos are 25fps

        frames = []
        for i, im in enumerate(video):
            if i % jump == 0:
                frames.append(im)

        frames = np.stack(frames)
        frames = frames.transpose(0, 3, 1, 2)

        frames = np.divide(frames, 255)

        frames = torch.from_numpy(frames)

        label = self.samples.iloc[idx, 1]
        if label == 'client':
            label = np.zeros((frames.shape[0], 1))
        else:
            label = np.ones((frames.shape[0], 1))
        label = torch.from_numpy(label)
        sample = {'image': frames, 'label': label}
        return sample
