from __future__ import print_function, division

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import json


FEATURE_FILE = '/home/rsun6573/work/pd_work/res_net_50_features/MJFF_{0}/{1}.csv'


class FeatureNomalise(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        pose, label = sample['pose'], sample['label']
        pose = pose/self.output_size
        return {'pose': pose, 'label': label}


class Rn50Dataset(Dataset):
    """Pose dataset."""

    def __init__(self, csv_file, fold_id, is_training, transform=None):
        self.video_df = self.get_video_df(csv_file, fold_id, is_training)
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
            file_name = FEATURE_FILE.format(video_id, file_id,)
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
    def get_video_df(csv_file, fold_id, is_training):
        df = pd.read_csv(csv_file)
        if is_training:
            df = df[df['group_id'] != fold_id]
        else:
            df = df[df['group_id'] == fold_id]

        df_1 = df[df['label'] == 1]
        df_0 = df[df['label'] == 0]
        result = df_1.groupby('video_id').agg('count').reset_index()

        dfs = []
        remainder = 0
        for index, row in result.iterrows():
            video_id = row['video_id']
            cnt = row['label']
            df_tmp = df_0[df_0['video_id'] == video_id]

            if remainder > 0:
                cnt = cnt + remainder
                remainder = 0

            try:
                # random_state=1 will give the same random samples
                df_tmp = df_tmp.sample(n=cnt, random_state=1)
            except:
                print('video:', video_id, row['subject_id'])
                remainder = cnt - df_tmp.shape[0]

            dfs.append(df_tmp)
        df_0 = pd.concat(dfs)
        df_all = pd.concat([df_0, df_1]).reset_index().drop(['index'], axis=1)
        return df_all

    @staticmethod
    def _read_pose(pose_file):
        with open(pose_file) as f:
            data = json.load(f)

        pose = []
        people = data.get('people')

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
