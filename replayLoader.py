import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import imageio
import random
from tqdm import tqdm


##### Fixing Seed #################
seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))
##############################



class replayLoader(Dataset):

    def __init__(self, csv_file, root_dir, frames_per_video):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
        """
        reader = pd.read_csv(csv_file)
        self.samples = reader
        self.root_dir = root_dir
        self.frames_per_video = frames_per_video

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.samples.iloc[idx, 0])
        video = imageio.get_reader(img_name, "ffmpeg")

        meta_data = video.get_meta_data()
        fps = meta_data['fps']
        duration = meta_data['duration']
        n_frames = np.ceil(duration*fps)
        hop = int(n_frames/self.frames_per_video)

        frames = []
        for i, im in enumerate(video):
            if i % hop == 0:
                frames.append(im)

        frames = np.stack(frames)
        frames = frames.transpose(0, 3, 1, 2)

        frames = np.divide(frames, 255)

        frames = torch.from_numpy(frames)

        label = self.samples.iloc[idx, 1]
        if label == 'client':
            label = np.array(0)
        else:
            label = np.array(1)
        label = torch.from_numpy(label)
        sample = {'image': frames, 'label': label}
        return sample

def run(csv_file_tr,root_dir_tr,frames_ps,csv_file_te,root_dir_te,batch_size=4,n_wor=4):
    data_train= replayLoader(csv_file_tr,root_dir_tr,frames_ps)
    data_test= replayLoader(csv_file_te,root_dir_te,frames_ps)
    dataloader_tr=DataLoader(data_train,batch_size,shuffle=True,num_workers=n_wor,worker_init_fn=_init_fn)
    dataloader_te=DataLoader(data_test,batch_size,shuffle=False,num_workers=n_wor,worker_init_fn=_init_fn)
    data= {'train':dataloader_tr,'test':dataloader_te}
    return data         

if __name__=="__main__":
    
    csv_file_tr='dataset/dataset_replay/train/data.csv'
    csv_file_te='dataset/dataset_replay/test/data.csv'
    root_dir_tr='dataset/dataset_replay/train/'
    root_dir_te='dataset/dataset_replay/test/'
    frames_ps=1


    data=run(csv_file_tr,root_dir_tr,frames_ps,csv_file_te,root_dir_te)
    phases=['train','test']
    for i in phases:
        for k in tqdm(data[i]):
            pass
            #import pdb;pdb.set_trace()


    





