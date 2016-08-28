import subprocess
import pickle
import argparse
from time import time

import uuid

import cv2
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_lowercase
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.preprocessing import scale

from edge import edge_detect, find_alphabet_contours, find_block
from sift import (load_SIFT_codebook, load_SIFT_bow, load_SIFT_w_augmentation,
                  extract_SIFT_w_augmentation, windows_get_bow_features)
# import progressbar

def get_alphabets_block_from_cam(cap):
    N = 8
    groups = []
    areas = []
    while len(groups) < N:
        ok, img = cap.read()
        if not ok:
            print('Q__q')
            continue
        try:
            corners, dst, area = find_block(img)
        except ValueError:
            print('T_T')
            continue
        groups.append((img, corners, dst))
        areas.append(area)

    areas = np.array(areas)
    m, s = np.mean(areas), np.std(areas)
    # print('mean: {}, std: {}'.format(m, s), areas)
    groups = [g for i, g in enumerate(groups) if abs(areas[i] - m) < s]
    areas = [g for i, g in enumerate(areas) if abs(areas[i] - m) < s]
    img, corners, dst = groups[0]
    radius = int(np.sqrt(areas[0] * img.shape[0] * img.shape[1]) / 10
                 + np.sqrt(img.shape[0]))
    print(radius)
    print('got {} from {} imgs'.format(len(groups), N))
    tmpimg = img.copy()
    cv2.fillPoly(tmpimg, pts=[corners.astype('int32')], color=(255, 255, 255))
    # mask = color_mask(img).astype('uint8') * 255
    alpha_contours = find_alphabet_contours(tmpimg)
    windows = []
    for c in alpha_contours:
        center = np.reshape(np.mean(c, 0), 2).astype(int)
        # cv2.circle(img, (center[0], center[1]), 10, (255, 0, 0), 6)
        windows.append([center-radius, center+radius])

    return img, corners, dst, windows

def get_alphabets_from_cam(cap):
    while True:
        ok, img = cap.read()
        if ok:
            break
        print('Q__q')

    radius = 128 / 2
    windows = []
    alpha_contours = find_alphabet_contours(img)
    for c in alpha_contours:
        center = np.reshape(np.mean(c, 0), 2).astype(int)
        # cv2.circle(img, (center[0], center[1]), 10, (255, 0, 0), 6)
        windows.append([center-radius, center+radius])

    return img, None, None, windows


def run_cnn():
    cap = cv2.VideoCapture(0)
    torch_process = subprocess.Popen(
        ['th', 'classify.lua', 'maomao.th', 'sis.npy'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    while True:
        try:
            img, corners, dst, windows = get_alphabets_from_cam(cap)
            patches = [
                img[int(rec[0][1]):int(rec[1][1]),
                    int(rec[0][0]):int(rec[1][0])]
                for rec in windows
            ]
            for i, p in enumerate(patches):
                print(p.shape)
                # cv2.imshow( 'patch{}'.format(i), p)
                # cv2.waitKey(10)
                if p.shape == (128, 128, 3):
                    cv2.imwrite('./tmpimg/{}.jpg'.format(uuid.uuid4().hex), p)

            # big_array = np.array([ p[:,:,::-1] / 255. for p in patches ])
            # if not big_array.size:
                # continue
            # np.save('sis.npy', big_array)
            # torch_process.stdin.write(b'1\n')
            # out = torch_process.stdout.readline()
            # out = [int(x) for x in out.strip().split()]
            # print(out)
            break
        except KeyboardInterrupt:
            break
    torch_process.terminate()

def run_svm(modelname):
    mode = 'normal'
    dim = 1024
    codebook = load_SIFT_codebook(mode, dim)
    codebook_kdtree = KDTree(codebook)
    with open(modelname, 'rb') as f:
        clf = pickle.load(f)
    # clf = train_SVM()

    cap = cv2.VideoCapture(0)
    while True:
        N = 8
        groups = []
        areas = []
        while len(groups) < N:
            ok, img = cap.read()
            if not ok:
                print('Q__q')
                continue
            try:
                corners, dst, area = find_block(img)
            except ValueError:
                print('T_T')
                continue
            groups.append((img, corners, dst))
            areas.append(area)

        areas = np.array(areas)
        m, s = np.mean(areas), np.std(areas)
        # rint('mean: {}, std: {}'.format(m, s), areas)
        groups = [g for i, g in enumerate(groups) if abs(areas[i] - m) < s]
        areas = [g for i, g in enumerate(areas) if abs(areas[i] - m) < s]
        img, corners, dst = groups[0]
        radius = int(np.sqrt(areas[0] * img.shape[0] * img.shape[1]) / 10
                     + np.sqrt(img.shape[0]))
        print(radius)
        print('got {} from {} imgs'.format(len(groups), N))

        cv2.fillPoly(img, pts=[corners.astype('int32')], color=(255, 255, 255))
        # mask = color_mask(img).astype('uint8') * 255
        alpha_contours = find_alphabet_contours(img)
        windows = []
        for c in alpha_contours:
            center = np.reshape(np.mean(c, 0), 2).astype(int)
            # cv2.circle(img, (center[0], center[1]), 10, (255, 0, 0), 6)
            windows.append([center-radius, center+radius])
            print(windows)

        feats = windows_get_bow_features(img, windows, dim, codebook_kdtree,
                                         mode)

        klasses = clf.predict(feats)
        print(klasses)
        best_window_idx = 0
        best_pred = klasses[0]
        # prob = clf.predict_proba(feats)
        # max_prob = np.max(prob, axis=1)
        # pred = np.argmax(prob, axis=1)

        # best_window_idx = np.argmax(max_prob)
        # best_pred = pred[best_window_idx]
        # print(prob)
        # print(pred)
        for i, rec in enumerate(windows):
            if i == best_window_idx:
                cv2.rectangle(img, tuple(rec[0]), tuple(rec[1]), (255, 0, 0), 3)
        print('Prediction: %s'%ascii_lowercase[best_pred])
        # cv2.drawContours(img, alpha_contours, -1, (255, 0, 0), 5)

        cv2.imshow('test', img)
        cv2.waitKey(10)

def accuracy(output, label):
    return np.sum(output == label) / np.shape(output)

def train_SVM(modelname):
    start_time = time()
    mode = 'normal'
    dim = 1024
    train_feats, train_labs, dev_feats, dev_labs = load_SIFT_w_augmentation(mode, dim)
    # train_feats, train_labs, dev_feats, dev_labs = load_SIFT_bow(mode, dim)
    # train_feats = scale(train_feats)
    # dev_feats = scale(dev_feats)
    print('Data loaded. Time: %.4fs' % (time() - start_time))

    best_acc = -1
    best_clf = None
    best_c = None
    print('Start training...')
    for c in range(700, 701, 25):
        clf = SVC(C=c, probability=False)
        clf.fit(train_feats, train_labs)
        dev_pred = clf.predict(dev_feats)
        acc = accuracy(dev_pred, dev_labs)
        print('Training C = %d, accuracy = %.4f' % (c, acc))
        if acc > best_acc:
            best_clf = clf
            best_acc = acc
            best_c = c
    print('Training finished. Time: %.4fs'%(time() - start_time))
    # best C: 350(256dim), 670(512dim), 1200(1024dim)
    print('Best C: %d' % best_c)
    print('Best accuracy: %.4f' % best_acc)
    with open(modelname, 'wb') as f:
        pickle.dump(best_clf, f)
    print('model saved at {}'.format(modelname))
    print('checking...')
    with open(modelname, 'rb') as f:
        clf = pickle.load(f)
    dev_pred = clf.predict(dev_feats)
    acc = accuracy(dev_pred, dev_labs)
    print('accuracy = %.4f' % (acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--extract', action='store_true')
    args = parser.parse_args()
    modelname = args.model
    if not modelname:
        modelname = 'svc_best.pickle'
    if args.extract:
        extract_SIFT_w_augmentation(mode='normal', dim=1024)
        # extract_SIFT_bow(mode='normal', codebook_num=1024,
                         # codebook_train_num=300)
    if args.train:
        train_SVM(modelname)
    else:
        # run_svm(modelname)
        run_cnn()
