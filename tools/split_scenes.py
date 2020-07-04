import os
import numpy as np
import cv2
import pickle
from ..params import datasetargs


'''split video scenes.
ref: http://www.c-s-a.org.cn/csa/ch/reader/create_pdf.aspx?file_no=20090841&flag=1&year_id=8&quarter_id='''

def hsv_quant(img):
    """quantilize hsv color space.
    
    Args:
        img: np.uint8 bgr image whose shape is (h, w, 3).

    Returns:
        np.uint8 (h, w) array.
    """
    img = np.float32(img) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv)

    # h
    h[h >= 316] = 0
    h[h < 20] = 0
    h[(h >= 20) & (h < 40)] = 1
    h[(h >= 40) & (h < 75)] = 2
    h[(h >= 75) & (h < 155)] = 3
    h[(h >= 155) & (h < 190)] = 4
    h[(h >= 190) & (h < 270)] = 5
    h[(h >= 270) & (h < 295)] = 6
    h[(h >= 295) & (h < 316)] = 7

    # s
    s[(s >= 0) & (s <= 0.2)] = 0
    s[(s > 0.2) & (s <= 0.7)] = 1
    s[(s > 0.7) & (s <= 1)] = 2

    # v
    v[(v >= 0) & (v <= 0.2)] = 0
    v[(v > 0.2) & (v <= 0.7)] = 1
    v[(v > 0.7) & (v <= 1)] = 2

    L = 9*h + 3*s + v
    return np.uint8(L)

def calc_hist(img):
    """calculate historam of one channel np.uint8 array."""
    hist = np.squeeze(cv2.calcHist([img], [0], np.ones(img.shape[:2], np.uint8), [72], [0.0, 71.0]))
    return hist

def hist_simil(hist1, hist2):
    """calculate similarities of histogram1 and histogram2."""
    hist1 = np.float32(hist1) / np.sum(hist1)
    hist2 = np.float32(hist2) / np.sum(hist2)

    simil = np.sum(np.minimum(hist1, hist2))

    return simil

def split_scenes(frame_names, frame_dir, thres=0.6):
    """split consecutive frames to scenes.
    
    Args:
        frame_names: a list of consecutive frame names.
        frame_dir: directory where frames exists.
        thres: if similarity is less than thres, scenes will be split.

    Returns:
        scenes: a nested list:[[frame1, frame2], [frame3,frame4], [...], ...]
                              [scene1,           scene2,          scene3, ...]
    """
    def _img_hist(path):
        '''calculate histogram of quantilized image.'''
        img = cv2.imread(path)
        h, w = img.shape[:2]
        # resize to reduce computational cost.
        img = cv2.resize(img, (int(w*0.25), int(h*0.25))) 
        hsv = hsv_quant(img)
        hist = calc_hist(hsv)
        return hist
    
    frame_names = sorted(frame_names)
    scenes = []

    onescene = []

    for i in range(len(frame_names)):
        if len(onescene) == 0:
            onescene.append(frame_names[i])
            lasthist = _img_hist(os.path.join(frame_dir, frame_names[i]))
        else:
            currhist = _img_hist(os.path.join(frame_dir, frame_names[i]))
            simi = hist_simil(lasthist, currhist)
            if simi < thres:
                scenes.append(onescene)
                onescene = []
            onescene.append(frame_names[i])
            lasthist = currhist

    scenes.append(onescene)
    return scenes

def scenes_to_5_frames_train_data(scenes):
    '''split scenes to train data, a piece of train data contains
    5 consecutive frames.
    
    Args:
        scenes: a list of scenes: [scene1, scene2, ...]

    Returns:
        data: a list of train data: [train data1, train data2, ...]
            each element contains 5 consecutive frames.
    '''

    def _expand_scene_to_at_least_3_frames(scene):
        num = len(scene)
        if num > 3:
            return scene
        if num == 1:
            return [scene[0], scene[0], scene[0]]
        if num == 2:
            return [scene[1]] + scene
        return []

    train_data = []
    for c in scenes:
        c = _expand_scene_to_at_least_3_frames(c)
        c = c[1:3][::-1] + c + c[-3:-1][::-1]
        num = len(c)
        for i in range(2, num-2):
            train_data.append(c[i-2:i+3])
    
    return train_data


if __name__ == '__main__':
    framesdir = os.path.join(datasetargs.image_dir, datasetargs.damage_sub_dir)
    respath = datasetargs.pkl_path
    
    frame_names = os.listdir(framesdir)
    scenes = split_scenes(frame_names, framesdir)
    traindata = scenes_to_5_frames_train_data(scenes)

    with open(traindata, 'wb') as f:
        pickle.dump(traindata, f)
