import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, BatchSampler, WeightedRandomSampler
import imageio
import random
from skimage.transform import resize
from tqdm import tqdm

##### Fixing Seed #################
seed = 10
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

def shuffle_slice(a, start, stop):
    i = start
    while (i < stop - 1):
        idx = random.randrange(i, stop)
        a[i], a[idx] = a[idx], a[i]
        i += 1


class OuluLoader(Dataset):

    def __init__(self, csv_file, root_dir, frames_per_video):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
        """
        reader = pd.read_csv(csv_file)
        self.all_labels = np.array(reader[reader.keys()[-1]])
        self.ind_imposters = np.where(self.all_labels == 'imposter')[0]
        self.ind_clients = np.where(self.all_labels == 'client')[0]

        # import pdb;pdb.set_trace()
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
        n_frames = np.ceil(duration * fps)
        hop = int(n_frames / self.frames_per_video)
        place = np.linspace(2, n_frames - 2, self.frames_per_video).astype('int')

        frames = []
        for i, im in enumerate(video):
            # if i % hop == 0:
            if i in place:
                frames.append(resize(im, (108, 192))

        frames = np.stack(frames)
        # import pdb;pdb.set_trace()
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


class Datasampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size):

        self.imposter_ind = data.ind_imposters
        self.client_ind = data.ind_clients
        # self.indices=torch.randperm(self.real_num)
        self.iter = []
        self.batch_size = batch_size

    def __iter__(self):

        self.iter = []
        if len(self.imposter_ind) == len(self.client_ind):
            ###### Equal number of samples will be distributed in evenly spaced minibatch ########
            self.imposter_rand = torch.randperm(len(self.imposter_ind))
            self.client_rand = torch.randperm(len(self.client_ind))
            max_len = len(self.imposter_ind)
            # import pdb;pdb.set_trace()
            for ind in range(max_len):
                ind_im = self.imposter_rand[ind]
                ind_cl = self.client_rand[ind]
                self.iter.append(int(self.imposter_ind[ind_im]))
                self.iter.append(int(self.client_ind[ind_cl]))
                r_ind = len(self.iter)
                if ((r_ind) % self.batch_size) == 0:
                    shuffle_slice(self.iter, max(0, r_ind - self.batch_size), r_ind)
            # import pdb;pdb.set_trace()
            return (self.iter[i] for i in range(len(self.iter)))
        else:
            ###### Unequal number of samples will be distributed in minibatch by their inverse of their probability of being present in a class ########
            class_sample_count = [len(self.client_ind), len(self.imposter_ind)]
            weights = 1 / torch.Tensor(class_sample_count)
            weights = weights.double()
            target = torch.cat((torch.ones(len(self.imposter_ind), dtype=torch.long),
                                torch.zeros(len(self.client_ind), dtype=torch.long)))
            samples_weight = torch.tensor([weights[t] for t in target])
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            # import pdb;pdb.set_trace()
            self.iter2 = list(sampler)
            # print( self.iter2)
            return (self.iter2[i] for i in range(len(self.iter2)))

    def __len__(self):
        return len(self.iter)


def run(csv_file_tr, root_dir_tr, frames_ps, csv_file_te, root_dir_te, batch_size=4, n_wor=0):
    data_train = OuluLoader(csv_file_tr, root_dir_tr, frames_ps)
    data_test = OuluLoader(csv_file_te, root_dir_te, frames_ps)
    sampler_tr = Datasampler(data_train, batch_size)

    # dataloader_tr=DataLoader(data_train,batch_size,shuffle=False,drop_last=True,sampler=sampler_tr,num_workers=n_wor,worker_init_fn=_init_fn)
    dataloader_tr = DataLoader(data_train, batch_size, shuffle=True, drop_last=True, num_workers=n_wor,
                               worker_init_fn=_init_fn)
    dataloader_te = DataLoader(data_test, batch_size, shuffle=False, num_workers=n_wor, worker_init_fn=_init_fn)
    data = {'train': dataloader_tr, 'test': dataloader_te}
    return data


if __name__ == "__main__":

    csv_file_tr = 'OULU.csv'
    csv_file_te = 'dataset/dataset_replay/test/data.csv'
    root_dir_tr = 'C:\\Users\\Raphael\\Downloads\\Train_files'
    root_dir_te = 'dataset/dataset_replay/test/'
    frames_ps = 10

    data = run(csv_file_tr, root_dir_tr, frames_ps, csv_file_te, root_dir_te)
    phases = ['train', 'test']
    labels = []
    for i in phases:
        for k in tqdm(data[i]):
            labels.append(k['label'])
            pass
    import pdb;

    pdb.set_trace()
