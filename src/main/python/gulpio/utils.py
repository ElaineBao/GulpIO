#!/usr/bin/env python

import os
import sh
import random
import cv2
import numpy as np
import shutil
import glob
from contextlib import contextmanager
from gulpio.fileio import GulpDirectory

###############################################################################
#                                Helper Functions                             #
###############################################################################


class FFMPEGNotFound(Exception):
    pass


def check_ffmpeg_exists():
    return os.system('ffmpeg -version > /dev/null') == 0


@contextmanager
def temp_dir_for_bursting(shm_dir_path='/dev/shm'):
    hash_str = str(random.getrandbits(128))
    temp_dir = os.path.join(shm_dir_path, hash_str)
    os.makedirs(temp_dir)  # creates error if paths conflict (unlikely)
    yield temp_dir
    shutil.rmtree(temp_dir)


def burst_frames_to_shm(vid_path, temp_burst_dir, image_ext, frame_rate=None):
    """
    - To burst frames in a temporary directory in shared memory.
    - Directory name is chosen as random 128 bits so as to avoid clash
      during parallelization
    - Returns path to directory containing frames for the specific video
    """
    target_mask = os.path.join(temp_burst_dir, '%04d{}'.format(image_ext))
    if not check_ffmpeg_exists():
        raise FFMPEGNotFound()
    try:
        ffmpeg_args = [
            '-i', vid_path,
            '-q:v', str(2),
            '-f', 'image2',
            target_mask,
        ]
        if frame_rate:
            ffmpeg_args.insert(2, '-r')
            ffmpeg_args.insert(3, frame_rate)
        sh.ffmpeg(*ffmpeg_args)
    except Exception as e:
        print(repr(e))


def burst_video_into_frames(vid_path, temp_burst_dir, image_ext='.jpg', frame_rate=None):
    burst_frames_to_shm(vid_path, temp_burst_dir, image_ext, frame_rate=frame_rate)
    return find_images_in_folder(temp_burst_dir, format_=image_ext)


def burst_flows_to_shm(vid_path, alg_type, temp_burst_dir, image_ext, flow_size):
    try:
        cap = cv2.VideoCapture(vid_path)
        if alg_type == 'farn':
            optical_flow_hdl = cv2.FarnebackOpticalFlow_create(pyrScale=0.702, numLevels=5, winSize=10,
                                                               numIters=2, polyN=5, polySigma=1.1,
                                                               flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        elif alg_type == 'tvl1':
            optical_flow_hdl = cv2.DualTVL1OpticalFlow_create()
        else:
            raise NotImplementedError('optical flow algorithm {} is not implemented.'.format(alg_type))

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        prvs = resize_by_short_edge(prvs, flow_size)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        idx = 1

        while ret:
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            next = resize_by_short_edge(next, flow_size)

            flow = optical_flow_hdl.calc(prvs, next, None)

            """
			flow_x = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
			flow_y = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
			flow_x = flow_x.astype('uint8')
			flow_y = flow_y.astype('uint8')
			video_flow_list.append([flow_x,flow_y])
			"""
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(temp_burst_dir, '{:04d}{}'.format(idx,image_ext)), bgr)

            prvs = next
            idx += 1
    except Exception as e:
        print(repr(e))

def burst_video_into_flows(vid_path, alg_type, temp_burst_dir, image_ext='.png', flow_size=-1):
    burst_frames_to_shm(vid_path, temp_burst_dir, flow_size)
    return find_images_in_folder(temp_burst_dir, format_=image_ext)


class ImageNotFound(Exception):
    pass


class DuplicateIdException(Exception):
    pass


def resize_images(imgs, img_size=-1):
    for img in imgs:
        img_path = img
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ImageNotFound("Image is  None from path:{}".format(img_path))
        if img_size > 0:
            img = resize_by_short_edge(img, img_size)
        yield img


def resize_by_short_edge(img, size):
    if isinstance(img, str):
        img_path = img
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ImageNotFound("Image read None from path ", img_path)
    if size < 1:
        return img
    h, w = img.shape[0], img.shape[1]
    if h < w:
        scale = w / float(h)
        new_width = int(size * scale)
        img = cv2.resize(img, (new_width, size))
    else:
        scale = h / float(w)
        new_height = int(size * scale)
        img = cv2.resize(img, (size, new_height))
    return img


def remove_entries_with_duplicate_ids(output_directory, meta_dict):
    meta_dict = _remove_duplicates_in_metadict(meta_dict)
    gulp_directory = GulpDirectory(output_directory)
    existing_ids = list(gulp_directory.merged_meta_dict.keys())
    # this assumes no duplicates in existing_ids
    new_meta = []
    for meta_info in meta_dict:
        if str(meta_info['id']) in existing_ids:
            print('Id {} already in GulpDirectory, I skip it!'
                  .format(meta_info['id']))
        else:
            new_meta.append(meta_info)
    if len(new_meta) == 0:
        print("no items to add... Abort")
        raise DuplicateIdException
    return new_meta


def _remove_duplicates_in_metadict(meta_dict):
    ids = list(enumerate(map(lambda d: d['id'], meta_dict)))
    if len(set(map(lambda d: d[1], ids))) == len(ids):
        return meta_dict
    else:
        new_meta = []
        seen_id = []
        for index, id_ in ids:
            if id_ not in seen_id:
                new_meta.append(meta_dict[index])
                seen_id.append(id_)
            else:
                print('Id {} more than once in json file, I skip it!'
                      .format(id_))
        return new_meta


###############################################################################
#                       Helper Functions for input iterator                   #
###############################################################################

def find_images_in_folder(folder, format_='.jpg'):
    files = glob.glob('{}/*.{}'.format(folder, format_))
    return sorted(files)


def get_single_video_path(folder_name, format_='mp4'):
    video_filenames = glob.glob("{}/*.{}".format(folder_name, format_))
    assert len(video_filenames) == 1
    return video_filenames[0]
