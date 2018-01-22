import numpy as np
import sys
import os
import csv
from scipy import misc
import scipy.stats as stats
import random
import pickle

hair_color = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
    'green hair', 'red hair', 'purple hair', 'pink hair',
    'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_color = ['gray eyes', 'black eyes', 'orange eyes',
    'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
    'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


def make_one_hot(hair, eye):

    eyes_hot = np.zeros([len(eye_color)])
    eyes_hot[eye] = 1
    hair_hot = np.zeros([len(hair_color)])
    hair_hot[hair] = 1
    tag_vec = np.concatenate((eyes_hot, hair_hot))

    return tag_vec


def load_test(test_path, hair_map, eye_map):

    test = []
    with open(test_path, 'r') as f:

        for line in f.readlines():
            hair = 0
            eye = 0
            if line == '\n':
                break
            line = line.strip().split(',')[1]
            p = line.split(' ')
            p1 = ' '.join(p[:2]).strip()
            p2 = ' '.join(p[-2:]).strip()
        
            if p1 in hair_map:
                hair = hair_map[p1]
            elif p2 in hair_map:
                hair = hair_map[p2]
            
            if p1 in eye_map:
                eye = eye_map[p1]
            elif p2 in eye_map:
                eye = eye_map[p2]

            test.append(make_one_hot(hair, eye))
    
    return test

 
def dump_img(img_dir, img_feats, test):

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        path = os.path.join(img_dir, 'sample_{}_{}.jpg'.format(test, idx+1))
        misc.imsave(path, img_feat)



def preprocessing(preproc_dir, img_dir, tag_path, eye_map, hair_map):

    attrib_tags = [] 
    img_feat = []
    img_size = 96
    resieze = int(96*1.15)

    with open(tag_path, 'r') as f:
        for idx, row in enumerate(csv.reader(f)):

            tags = row[1].split('\t')
            hair = 'unk'
            eyes = 'unk'
            has_hair = False
            has_eye = False
            skip_hair = False
            skip_eye = False
            skip = False

            for t in tags:
                if t != '':
                    tag = t.split(':')[0].strip()

                    if tag == 'bicolored eyes':
                        print(tag)
                        skip = True
                        break

                    if tag in eye_map:

                        if has_eye:
                            skip_hair = True
                        
                        eyes = tag
                        has_eye = True

                    elif tag in hair_map:
                        if has_hair:
                            skip_eye = True

                        hair = tag
                        has_hair = True

            if skip_hair:
                hair = 'unk'

            if skip_eye:
                eyes = 'unk'


            if eyes == 'unk' or hair == 'unk':
                skip = True

            if skip:
                continue

            hair_idx = hair_map[hair]
            eyes_idx = eye_map[eyes]


            img_path = os.path.join(img_dir, '{}.jpg'.format(idx))
            feat = misc.imread(img_path)
            feat = misc.imresize(feat, [img_size, img_size, 3])
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat)

            m_feat = np.fliplr(feat)
            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(m_feat)

            feat_p5 = misc.imrotate(feat, 5)
            feat_p5 = misc.imresize(feat_p5, [resieze , resieze, 3])
            feat_p5 = crop_center(feat_p5, img_size,img_size)

            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_p5)

            feat_m5 = misc.imrotate(feat, -5)
            feat_m5 = misc.imresize(feat_m5, [resieze, resieze, 3])
            feat_m5 = crop_center(feat_m5, img_size,img_size)

            attrib_tags.append([hair_idx, eyes_idx])
            img_feat.append(feat_m5)

    img_feat = np.array(img_feat)

    pickle.dump(img_feat, open(os.path.join(preproc_dir, "img_feat_96.dat"), 'wb'))
    pickle.dump(attrib_tags, open(os.path.join(preproc_dir, "tags.dat"), 'wb'))

    return img_feat, attrib_tags


class Data(object):
    def __init__(self, img_feat, tags, z_dim):
        self.z_dim = z_dim
        self.z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)
        self.length = len(tags)
        self.current = 0
        self.img_feat = img_feat
        self.tags = np.array(tags)
        self.epoch = 0
        
        self.hair_idx = np.array([idx for idx in set(self.tags[:,0])])
        self.eyes_idx = np.array([idx for idx in set(self.tags[:,1])])

        self.h_len = len(self.hair_idx)
        self.e_len = len(self.eyes_idx)
        self.test = [[1,2],[1,3]]
        #o b, a p, g y, g g
        self.test_tags_idx = self.gen_test_hot()
        self.fixed_z = self.next_noise_batch(len(self.test_tags_idx), z_dim)

    def gen_test_hot(self):
        test_hot = []
        for tag in self.test:
            test_hot.append(make_one_hot(tag[0], tag[1]))

        return np.array(test_hot)

    def load_eval(self,test_path, hair_map, eye_map):
        
        self.test_tags_idx = load_test(test_path, hair_map, eye_map)
        self.fixed_z = self.next_noise_batch(len(self.test_tags_idx), self.z_dim)


    def next_data_batch(self, size, neg_sample=False):
        self.size = size
        if self.current == 0:
            
            self.epoch += 1
            idx = np.random.permutation(np.arange(self.length))

            self.img_feat = self.img_feat[idx]
            self.tags = self.tags[idx]
            idx = np.random.permutation(np.arange(self.length))

        if self.current + size < self.length:

            img, tags = self.img_feat[self.current:self.current+size], self.tags[self.current:self.current+size]
            self.current += size

        else:
        
            img, tags = self.img_feat[-size:], self.tags[-size:]
            self.current = 0
            self.size = len(img)

        tag_one_hot = []

        for t in tags:
            tag_one_hot.append(make_one_hot(t[0]-1, t[1]-1))

        return img, np.array(tag_one_hot)

    def next_noise_batch(self, size, dim):
        return self.z_sampler.rvs([size, dim]) #np.random.uniform(-1.0, 1.0, [size, dim])

def dump_img(img_dir, img_feats, iters, img_size = 96):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)

    for idx, img_feat in enumerate(img_feats):
        path = os.path.join(img_dir, 'iters_{}_test_{}.jpg'.format(iters, idx))
        img_feat =  misc.imresize(img_feat, [img_size, img_size, 3])
        misc.imsave(path, img_feat)

