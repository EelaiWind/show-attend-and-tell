from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

DATA_BASE_LABEL = ('train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'val', 'test')
ALL_TRAIN_DATA = 'train'
IMAGE_ID = 'image_id'
CAPTION = 'caption'
FILE_NAME = 'file_name'

def _process_caption_data(caption_file, image_dir, max_length, is_train_data):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image['id']: image[FILE_NAME] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation[IMAGE_ID]
        annotation[FILE_NAME] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]
    
    all_data = []
    if is_train_data == True:
        segment = (0, 84997, 169996, 255000, 339997, 414113)
        data.sort(key=lambda x: x[IMAGE_ID])
        all_data.append(data[segment[0]:segment[1]])
        all_data.append(data[segment[1]:segment[2]])
        all_data.append(data[segment[2]:segment[3]])
        all_data.append(data[segment[3]:segment[4]])
        all_data.append(data[segment[4]:segment[5]])
    else:
        all_data.append(data)

    # convert to pandas dataframe (for later visualization or debugging)
    all_caption_data = []
    for data in all_data:
        caption_data = pd.DataFrame.from_dict(data)
        del caption_data['id']
        caption_data.sort_values(by=IMAGE_ID, inplace=True)
        caption_data = caption_data.reset_index(drop=True)
        
        del_idx = []
        for i, caption in enumerate(caption_data[CAPTION]):
            caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
            caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
            caption = " ".join(caption.split())  # replace multiple spaces
            
            caption_data.set_value(i, CAPTION, caption.lower())
            if len(caption.split(" ")) > max_length:
                del_idx.append(i)
        
        # delete captions if size is larger than max_length
        print "The number of captions before deletion: %d" %len(caption_data)
        caption_data = caption_data.drop(caption_data.index[del_idx])
        caption_data = caption_data.reset_index(drop=True)
        print "The number of captions after deletion: %d" %len(caption_data)
        all_caption_data.append(caption_data)
    
    return all_caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations[CAPTION]):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations[CAPTION]):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations[IMAGE_ID]
    file_names = annotations[FILE_NAME]
    for image_id, file_name in zip(image_ids, file_names):
        if not image_id in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations[IMAGE_ID]
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # batch size for extracting feature vectors from vggnet.
    batch_size = 100
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    # vgg model path 
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    caption_file = 'data/annotations/captions_train2014.json'
    image_dir = 'image/%2014_resized/'

    # about 80000 images and 400000 captions for train dataset
    split_train_dataset = _process_caption_data(caption_file='data/annotations/captions_train2014.json',
                                          image_dir='image/train2014_resized/',
                                          max_length=max_length,
                                          is_train_data=True)
    
    train_dataset = _process_caption_data(caption_file='data/annotations/captions_train2014.json',
                                          image_dir='image/train2014_resized/',
                                          max_length=max_length,
                                          is_train_data=False)
    train_dataset = train_dataset[0]

    # about 40000 images and 200000 captions
    val_dataset = _process_caption_data(caption_file='data/annotations/captions_val2014.json',
                                        image_dir='image/val2014_resized/',
                                        max_length=max_length,
                                        is_train_data=False)
    
    # do not split test / validation dataset
    val_dataset = val_dataset[0]

    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_dataset))
    test_cutoff = int(0.2 * len(val_dataset))
    print 'Finished processing caption data'

    for i in xrange(len(split_train_dataset)):
        save_pickle(split_train_dataset[i], 'data/train_%d/train_%d.annotations.pkl' % (i, i) )
    save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    save_pickle(val_dataset[:val_cutoff], 'data/val/val.annotations.pkl')
    save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    split = 'train'
    annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))
    word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
    save_pickle(word_to_idx, './data/%s/word_to_idx.pkl' % split)    

    #for split in ['train', 'val', 'test']:
    for split in DATA_BASE_LABEL:
        annotations = load_pickle('./data/%s/%s.annotations.pkl' % (split, split))
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        print "[DEBUG] captions = ", captions.shape
        save_pickle(captions, './data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        print "[DEBUG] file_names = ", len(file_names)
        print "[DEBUG] id_to_idx = ", len(id_to_idx)
        save_pickle(file_names, './data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        print "[DEBUG] image_idxs = ", len(image_idxs)
        save_pickle(image_idxs, './data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = {}
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations[CAPTION], annotations[IMAGE_ID]):
            if not image_id in image_ids:
                image_ids[image_id] = 0
                i += 1
                feature_to_captions[i] = []
            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data/%s/%s.references.pkl' % (split, split))
        print "[DEBUG] feature_to_captions = ", len(feature_to_captions)
        print "Finished building %s caption dataset" %split

    # extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for split in DATA_BASE_LABEL:
            anno_path = './data/%s/%s.annotations.pkl' % (split, split)
            save_path = './data/%s/%s.features.hkl' % (split, split)
            annotations = load_pickle(anno_path)
            image_path = list(annotations[FILE_NAME].unique())
            n_examples = len(image_path)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d %s features.." % (end, split))

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))


if __name__ == "__main__":
    main()
