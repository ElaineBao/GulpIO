import os
import numpy as np
from numpy.random import randint
import json
from .fileio import GulpDirectory
from PIL import Image
import cv2

class GulpIOEmptyFolder(Exception):  # pragma: no cover
        pass


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

class GulpVideoDataset(object):

    def __init__(self, list_file, data_path, num_segments=3, test_mode=False,
                 transform=None):
        r"""Simple data loader for GulpIO format.

            Args:
                data_path (str): path to GulpIO dataset folder
        """
        self.list_file = list_file
        self.data_path = data_path
        self.num_segments = num_segments
        self.transform = transform
        self.test_mode = test_mode
        self._parse_list()

        self.gd = GulpDirectory(data_path)
        self.items = self.gd.merged_meta_dict

        self.num_chunks = self.gd.num_chunks

        if self.num_chunks == 0:
            raise(GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                    "of: ".format(data_path)))

        print(" > Found {} chunks".format(self.num_chunks))


    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]


    def _sample_indices(self, num_frames):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = num_frames // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, num_frames):
        if num_frames > self.num_segments:
            tick = num_frames / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        record = self.video_list[index]
        video_path = record.path
        label = record.label
        video_info = self.items[video_path]

        target_index = video_info['meta_data'][0]['label']
        assert label == target_index, "list label != chunk label"
        frames = video_info['frame_info']
        num_frames = len(frames)
        num_frames2 = video_info['meta_data'][0]['frame_num']
        assert num_frames == num_frames2, "frame_info len != chunk frame_num"
        # set number of necessary frames
        if not self.test_mode:
            segment_indices = self._sample_indices(num_frames)
        else:
            segment_indices = self._get_val_indices(num_frames)
        frames, meta = self.gd[video_path, segment_indices]
        pil_frames =[Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) for cv2_im in frames]

        if self.transform:
            pil_frames = self.transform(pil_frames)
        # format data to torch tensor

        return pil_frames, label

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.video_list)


class GulpImageDataset(object):

    def __init__(self, data_path, is_val=False, transform=None,
                 target_transform=None):
        r"""Simple image data loader for GulpIO format.

            Args:
                data_path (str): path to GulpIO dataset folder
                label_path (str): path to GulpIO label dictionary matching
            label ids to label names
                is_va (bool): sets the necessary augmention procedure.
                transform (object): set of augmentation steps defined by
            Compose(). Default is None.
                target_transform (func): performs preprocessing on labels if
            defined. Default is None.
        """

        self.gd = GulpDirectory(data_path)
        self.items = list(self.gd.merged_meta_dict.items())
        self.label2idx = json.load(open(os.path.join(data_path,
                                                     'label2idx.json')))
        self.num_chunks = self.gd.num_chunks

        if self.num_chunks == 0:
            raise(GulpIOEmptyFolder("Found 0 data binaries in subfolders " +
                                    "of: ".format(data_path)))

        print(" > Found {} chunks".format(self.num_chunks))
        self.data_path = data_path
        self.classes = self.label2idx.keys()
        self.transform = transform
        self.target_transform = target_transform
        self.is_val = is_val

    def __getitem__(self, index):
        """
        With the given video index, it fetches frames. This functions is called
        by Pytorch DataLoader threads. Each Dataloader thread loads a single
        batch by calling this function per instance.
        """
        item_id, item_info = self.items[index]

        target_name = item_info['meta_data'][0]['label']
        target_idx = self.label2idx[target_name]
        img_rec = item_info['frame_info']
        assert len(img_rec) == 1
        # set number of necessary frames
        img, meta = self.gd[item_id]
        img = img[0]
        # augmentation
        if self.transform:
            img = self.transform(img)
        return (img, target_idx)

    def __len__(self):
        """
        This is called by PyTorch dataloader to decide the size of the dataset.
        """
        return len(self.items)


if __name__ == '__main__':

    list_file = '../../../../bin/test-data.txt'
    gulp_folder = '../../../../bin/test-data-gulp/'
    print(list_file,gulp_folder)
    dataset = GulpVideoDataset(list_file,gulp_folder,num_segments=1)

    for item in dataset:
        print(item)
