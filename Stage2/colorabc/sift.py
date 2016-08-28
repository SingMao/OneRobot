import numpy as np
import cv2
import json
from string import ascii_lowercase
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KDTree

def split_dataset(letter_cnt):
    train_imgs = []
    train_labs = []
    dev_imgs = []
    dev_labs = []
    alph_idx = 0
    for c in ascii_lowercase:
        for i in range(2, letter_cnt[c]+1):
            img = cv2.imread('../alphabet_data/test/{}{}.jpg'.format(c, i))
            if np.random.random() < 0.8:
                train_imgs.append(img)
                train_labs.append(alph_idx)
            else:
                dev_imgs.append(img)
                dev_labs.append(alph_idx)
        alph_idx += 1

    return train_imgs, train_labs, dev_imgs, dev_labs

def get_bow_codebook(imgs, mode, codebook_num, kmeans_train_num):
    # get SIFT descriptors
    print('Obtaining SIFT descriptors...')
    i = 0
    sift = cv2.xfeatures2d.SIFT_create()
    sift_feats = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift_feats.append(get_sift_features(sift, img, mode))

    # obtain codebook and codebook_kdtree
    print('Performing k-Means...')
    permu = np.random.permutation(len(sift_feats))
    for i in range(kmeans_train_num):
        if i == 0:
            kmeans_train_dat = sift_feats[permu[i]]
        else:
            kmeans_train_dat = np.vstack([kmeans_train_dat, sift_feats[permu[i]]])
    if mode == 'dense':
        kmeans = MiniBatchKMeans(n_clusters=codebook_num, verbose=True)
    else:
        kmeans = KMeans(n_clusters=codebook_num, verbose=True)
    kmeans.fit(kmeans_train_dat)
    codebook = kmeans.cluster_centers_
    codebook_kdtree = KDTree(codebook)
    return codebook, codebook_kdtree

def get_bow_feature(sift_feats, codebook_num, codebook_kdtree):
    # get Bag-of-Words (BoW) features
    bow_feat = np.zeros(codebook_num)
    for i in range(sift_feats.shape[0]):
        dist, idx = codebook_kdtree.query(sift_feats[i:i+1,:], k=1)
        bow_feat[idx] += 1
    bow_feat = bow_feat / np.linalg.norm(bow_feat)
    return bow_feat

def windows_get_bow_features(img, windows, dim, codebook_kdtree, mode):
    # windows: [[ndarray, ndarray],[],[],...]
    sift = cv2.xfeatures2d.SIFT_create()
    bow_feats = []
    for rec in windows:
        patch = img[int(rec[0][1]):int(rec[1][1]),int(rec[0][0]):int(rec[1][0])]
        # while True:
            # try:
                # cv2.imshow('patch', patch)
                # cv2.waitKey(100)
            # except KeyboardInterrupt:
                # break

        sift_feats = np.vstack(filter(
            lambda x: x is not None,
            [get_sift_features(sift, patch[:,:,i], mode) for i in range(3)]
        ))
        color_mean = np.mean(patch, axis=(0, 1))
        bow_feat = np.hstack([get_bow_feature(sift_feats, dim, codebook_kdtree),
                              color_mean/np.linalg.norm(color_mean)])
        bow_feats.append(bow_feat)
    bow_feats = np.array(bow_feats)
    return bow_feats

def get_sift_features(sift, patch, mode):
    if mode == 'dense':
        step_size = 5
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, patch.shape[0], step_size)
                                            for x in range(0, patch.shape[1], step_size)]
        kp, des = sift.compute(patch, kp)
    else:
        kp, des = sift.detectAndCompute(patch, None)
    # lala = cv2.drawKeypoints(patch, kp, patch)
    # cv2.imshow('patch', lala)
    # cv2.waitKey(10)
    return des

def load_SIFT_codebook(mode, dim):
    npzfile = np.load('./extracted_features/{}_SIFT_BoW_codebook_{}d.npz'.format(mode, dim))
    return npzfile['codebook']

def extract_SIFT_bow(mode, codebook_num, codebook_train_num):
    # open alphabet count
    with open('../letter_cnt.json', 'r') as f:
        letter_cnt = json.loads(f.read())
    print(letter_cnt)

    # split data into training and dev set
    train_imgs, train_labs, dev_imgs, dev_labs = split_dataset(letter_cnt)
    print(len(train_labs))
    print(len(dev_labs))

    codebook, codebook_kdtree = get_bow_codebook(train_imgs, mode, codebook_num, codebook_train_num)

    # obtain Bag-of-Word features using SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    train_feats = []
    dev_feats = []
    for img in train_imgs:
        sift_feats = np.vstack([get_sift_features(sift, img[:,:,i], mode)
                                for i in range(3)])
        train_feats.append(get_bow_feature(sift_feats, codebook_num,
                                           codebook_kdtree))
    for img in dev_imgs:
        sift_feats = np.vstack([get_sift_features(sift, img[:,:,i], mode)
                                for i in range(3)])
        dev_feats.append(get_bow_feature(sift_feats, codebook_num,
                                         codebook_kdtree))

    print(len(train_feats))
    print(len(dev_feats))
    train_feats = np.array(train_feats)
    train_labs = np.array(train_labs)
    dev_feats = np.array(dev_feats)
    dev_labs = np.array(dev_labs)
    np.savez('./extracted_features/{}_SIFT_BoW_feats_{}d'.format(mode, codebook_num),
             train_feats=train_feats, train_labs=train_labs,
             dev_feats=dev_feats, dev_labs=dev_labs)
    np.savez('./extracted_features/{}_SIFT_BoW_codebook_{}d'.format(mode, codebook_num), codebook=codebook)

def extract_SIFT_w_augmentation(mode, dim):
    codebook = load_SIFT_codebook(mode, dim)
    codebook_kdtree = KDTree(codebook)

    with open('../letter_cnt.json', 'r') as f:
        letter_cnt = json.loads(f.read())

    train_imgs, train_labs, dev_imgs, dev_labs = split_dataset(letter_cnt)

    sift = cv2.xfeatures2d.SIFT_create()
    train_feats = []
    dev_feats = []
    # progress = ProgressBar(maxval=len(train_imgs)*100)
    # i = 0
    for img in train_imgs:
        sift_feats = np.vstack([get_sift_features(sift, img[:,:,i], mode)
                                for i in range(3)])
        color_mean = np.mean(img, axis=(0, 1))
        bow_feat = np.hstack([get_bow_feature(sift_feats, dim, codebook_kdtree),
                              color_mean/np.linalg.norm(color_mean)])
        train_feats.append(bow_feat)
        # for x in range(10):
            # for y in range(10):
                # sift_feats = get_sift_features(sift, img[6*y:6*y+162,6*x:6*x+162,:], mode)
                # train_feats.append(get_bow_feature(sift_feats, dim, codebook_kdtree))
                # # progress.update(i)
                # i += 1
                # print('\r%d'%i,end='')
    for img in dev_imgs:
        sift_feats = np.vstack([get_sift_features(sift, img[:,:,i], mode)
                                for i in range(3)])
        color_mean = np.mean(img, axis=(0, 1))
        bow_feat = np.hstack([get_bow_feature(sift_feats, dim, codebook_kdtree),
                              color_mean/np.linalg.norm(color_mean)])
        dev_feats.append(bow_feat)

    train_feats = np.array(train_feats)
    train_labs = np.array(train_labs)
    dev_feats = np.array(dev_feats)
    dev_labs = np.array(dev_labs)
    np.savez('./extracted_features/{}_aug_feats_{}d'.format(mode, dim),
             train_feats=train_feats, train_labs=train_labs,
             dev_feats=dev_feats, dev_labs=dev_labs)
    print(train_feats.shape[1])
    return train_feats, train_labs, dev_feats, dev_labs

def load_SIFT_bow(mode, dim):
    npzfile = np.load('./extracted_features/{}_SIFT_BoW_feats_{}d.npz'.format(mode, dim))
    train_feats = npzfile['train_feats']
    train_labs = npzfile['train_labs']
    dev_feats = npzfile['dev_feats']
    dev_labs = npzfile['dev_labs']
    return train_feats, train_labs, dev_feats, dev_labs

def load_SIFT_w_augmentation(mode, dim):
    npzfile = np.load('./extracted_features/{}_aug_feats_{}d.npz'.format(mode, dim))
    train_feats = npzfile['train_feats']
    train_labs = npzfile['train_labs']
    dev_feats = npzfile['dev_feats']
    dev_labs = npzfile['dev_labs']
    return train_feats, train_labs, dev_feats, dev_labs
