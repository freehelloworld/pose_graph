from __future__ import print_function, division

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import json


POSE_FILE = '~/Downloads/pose_data/MJFF_{0}/{1}_keypoints.json'


class PoseNomalise(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        pose, label = sample['pose'], sample['label']
        pose = pose/self.output_size
        return {'pose': pose, 'label': label}


class PoseDataset(Dataset):
    """Pose dataset."""

    def __init__(self, csv_file, transform=None):
        self.video_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        second_id = self.video_df.iloc[idx, 3]
        video_id = self.video_df.iloc[idx, 2]
        file_id = (second_id - 1) * 25 + 12
        try:
            file_name = POSE_FILE.format(video_id, file_id,)
            pose = self._read_pose(file_name)
            if len(pose) > 200:
                pose = pose[0:200]
            pose = pose + [0] * (200 - len(pose))
            pose_marks = np.array([pose])
            pose_marks = pose_marks.astype('float').reshape(-1, 2)
            sample = {'label': self.video_df.iloc[idx, 0], 'pose': pose_marks}

            if self.transform:
                sample = self.transform(sample)

            return sample
        except FileNotFoundError as err:
            #print(err)
            pose = np.random.rand(200)
            pose_marks = pose.astype('float').reshape(-1, 2)
            sample = {'label': 0, 'pose': pose_marks}
            return sample

    @staticmethod
    def _read_pose(pose_file):
        with open(pose_file) as f:
            data = json.load(f)

        pose = []
        people = data.get('people')
        #print('people', len(people))
        for person in people:
            points = person.get('pose_keypoints_2d')

            for idx, point in enumerate(points):
                if idx % 3 == 0:
                    pose.append(point)
                elif idx % 3 == 1:
                    pose.append(point)
                else:
                    pass
        return pose


# pose_data = PoseDataset('video_data1.csv')
#
# print(len(pose_data))
#
# dataloader = DataLoader(pose_data, batch_size=4,
#                         shuffle=True, num_workers=0)
#
# for i_batch, sample_batched in enumerate(dataloader):
#     pass
