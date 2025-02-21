import argparse
import os

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    descriptions = np.load(os.path.join(args.data_dir, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
    imgs_list = np.load(os.path.join(args.data_dir, 'img_cache4.npy'), encoding='latin1')
    followings_list = np.load(os.path.join(args.data_dir, 'following_cache4.npy'))
    train_ids, val_ids, test_ids = np.load(os.path.join(args.data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    train_ids = np.sort(train_ids) # 10191
    val_ids = np.sort(val_ids)  # 2334
    test_ids = np.sort(test_ids) # 2208

    if (args.use_subset):
        train_ids = train_ids[:len(train_ids)//4]
        print("Using a subset of training set, the size is : ", len(train_ids))
        #val_ids = val_ids[:1000]
        #test_ids = test_ids[:1000]

    f = h5py.File(args.save_path, "w")
    for subset, ids in {'train': train_ids, 'val': val_ids, 'test': test_ids}.items():
        length = len(ids)

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            img_paths = [str(imgs_list[item])[2:-1]] + [str(followings_list[item][i])[2:-1] for i in range(4)]
            imgs = [Image.open(os.path.join(args.data_dir, img_path)).convert('RGB') for img_path in img_paths]
            for j, img in enumerate(imgs):
                img = np.array(img).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
            tgt_img_ids = [str(img_path).replace('.png', '') for img_path in img_paths]
            txt = [descriptions[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones pororo file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='pororo data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    parser.add_argument('--use_subset', action='store_true', help='use subset of data')
    args = parser.parse_args()
    main(args)


# unzip original_datasets/pororo.zip -d original_datasets/

'''
nohup python data_script/pororo_hdf5.py --data_dir ./original_datasets/pororo_png --save_path ./processed_datasets/pororo.h5 > pororo.log 2>&1 &


python data_script/pororo_hdf5.py --data_dir ./original_datasets/pororo_png --save_path ./processed_subsets/pororo_1_4.h5 --use_subset

nohup command | tee output.log &
'''