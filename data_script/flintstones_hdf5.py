import argparse
import json
import os
import pickle

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def main(args):
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    # if (args.use_subset):
    #     train_ids = train_ids[:len(train_ids)// 8]
    #     print("Using a subset of training set, the size is : ", len(train_ids))
    #     #val_ids = val_ids[:1000]
    #     #test_ids = test_ids[:len(test_ids)// 10]
    #     test_ids = test_ids[:args.subset_size]

    f = h5py.File(args.save_path, "w")
    generated_datasets = defaultdict(list)

    if (args.test_only):  # only generate test set
        #print("Only generating test set, size is: {}".format(len(test_ids)))
        if (args.use_subset):
            test_ids = test_ids[:args.subset_size]
            print("Using a subset of test set, the size is : ", len(test_ids))
        generated_datasets = {'test': test_ids}
        
    
    else: # fully generate train, val, test set
        if (args.use_subset):
            train_ids = train_ids[:args.subset_size]
            print("Using a subset of training set, the size is : ", len(train_ids))
        generated_datasets = {'train': train_ids, 'val': val_ids, 'test': test_ids}


    for subset, ids in generated_datasets.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4] # exclude some samples
        length = len(ids)

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = [item] + followings[item]
            txt = list()
            for j, globalID in enumerate(globalIDs):
                img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))

                img = np.concatenate(img, axis=0).astype(np.uint8) # (640, 128, 3)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
                txt.append(descriptions[globalID])
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    parser.add_argument('--use_subset', action='store_true', help='use subset of data')
    parser.add_argument('--subset_size', type=int, help='subset size')
    parser.add_argument('--test_only', action='store_true', help='only test')

    args = parser.parse_args()

    # parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    # args = parser.parse_args()
    # args.data_dir = "./original_datasets/flintstones_data"
    # args.save_path = "./testing_sets/flintstones_testing_1.h5"
    # args.use_subset = True
    # args.subset_size = 3
    # args.test_only = True

    main(args)


# unzip original_datasets/flintstones_data.zip -d original_datasets

'''
python data_script/flintstones_hdf5.py --data_dir ./original_datasets/flintstones_data --save_path ./processed_datasets/flintstones.h5

python data_script/flintstones_hdf5.py --data_dir ./original_datasets/flintstones_data --save_path ./processed_subsets/flintstones_8.h5 --use_subset

python data_script/flintstones_hdf5.py --data_dir ./original_datasets/flintstones_data --save_path ./testing_sets/flintstones_testing_100.h5 --use_subset --subset_size 100 --test_only

'''
