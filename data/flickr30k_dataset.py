import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

import io
import cv2
import numpy as np

class flickr30k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', client=None):        
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_train.json'
        filename = 'flickr30k_train.json'

        self.client = client
        if self.client is not None:
            self.annotation = json.loads(client.get(os.path.join('s3://BucketName/ProjectName', ann_root, filename), enable_cache=True))
        else:
            download_url(url,ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if self.client is not None:
            image_path = os.path.join('s3://BucketName',self.image_root,ann['image'])      
            with io.BytesIO(self.client.get(image_path)) as f:
                image = Image.open(f).convert('RGB')   
            image = self.transform(image)   
        else:
            image_path = os.path.join(self.image_root,ann['image'])        
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        return image, caption, self.img_ids[ann['image_id']] 
    
    
class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, client=None):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        self.client = client
        if self.client is not None:
            self.annotation = json.loads(client.get(os.path.join('s3://BucketName/ProjectName', ann_root, filenames[split])))
        else:
            download_url(urls[split],ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.max_words = max_words
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]

        if self.client is not None:
            image_path = os.path.join('s3://BucketName',self.image_root, self.annotation[index]['image'])      
            with io.BytesIO(self.client.get(image_path)) as f:
                image = Image.open(f).convert('RGB')   
            image = self.transform(image)   
        else:
            image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
            image = Image.open(image_path).convert('RGB')    
            image = self.transform(image)  

        caption = pre_caption(ann['caption'][0], self.max_words) 
        return image, caption, index    