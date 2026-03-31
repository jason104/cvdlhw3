import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import torch
import copy
from PIL import Image
from glob import glob
import json
from torchvision.transforms import functional as F
import random
import glob


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        if target != None:
            return image, target
        else:
            return image

class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class ToTensor(object):

    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target
    

def get_transform(transType = 'valid'):
    if transType == 'train':
        return Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    else:
        return Compose([ToTensor()])
    # TODO
    # perform data augmentation
    # convert data type and value range
    # return transform


class SourceDataset(Dataset):

    def __init__(self, root: str, split: str = "org/train", transform=None, *args, **kwargs) -> None:
        super().__init__()
        self.root_dir = root
        self.anno_file = os.path.join(root, split + ".coco.json")
        self.transforms = transform
        self.coco = COCO(self.anno_file)
        with open(self.anno_file) as tmp:
            self.anno = json.load(tmp)
        # TODO

    def _load_image(self, index: int):
        a = self.anno['images'][index]
        #image_idx = a['id']
        cur_path = a['file_name']
        img_path = os.path.join(self.root_dir, cur_path)
        image = Image.open(img_path)
        return image
        # return image

    def _load_target(self, index: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(index))
        return target
        # return target

    def __getitem__(self, index: int):
        
        image = self._load_image(index)
        target = copy.deepcopy(self._load_target(index))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transforms is not None:
            image, boxes = self.transforms(image=image, target=boxes)
        
        #image = transformed['image']
        #boxes = transformed['bboxes']
        # xmin, ymin, w, h -> xmin, ymin, xmax, ymax
        new_boxes = []
        for box in boxes:
            xmin =  box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        if len(new_boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        #if new_boxes:
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
        
        return image, targ
        # TODO: make sure your image is scaled properly
        # return image and target

    def __len__(self) -> int:
        return len(self.anno['images'])
        # return the length of dataset


class TargetDataset(Dataset):

    def __init__(self, root: str, split: str = "fog/train", transform=None, *args, **kwargs) -> None:
        super().__init__()
        self.root_dir = os.path.join(root, split)
        #self.anno_file = os.path.join(root, split+"coco.json")
        self.transforms = transform
        file_list = os.listdir(self.root_dir)
        self.data_list = [x for x in file_list if x.endswith(".png")]
        #self.coco = COCO(self.anno_file)
        #with open(self.anno_file) as tmp:
        #    self.anno = json.load(tmp)
        # TODO

    def _load_image(self, index: int):
        img_path = os.path.join(self.root_dir, self.data_list[index])
        image = Image.open(img_path)
        return image
        # return image

    def __getitem__(self, index: int):
        image = self._load_image(index)
        if self.transforms is not None:
            image = self.transforms(image=image)
        return image
        # return image and target

    def __len__(self) -> int:
        return len(self.data_list)
        # return the length of dataset

class TestDataset(Dataset):

    def __init__(self, root: str, split: str = "fog/public_test", transform=None, *args, **kwargs) -> None:
        super().__init__()
        self.root_dir = os.path.join(root, split)
        #self.anno_file = os.path.join(root, split+"coco.json")
        self.transforms = transform
        self.data_list = glob.glob(os.path.join(root, '**/*.png'), recursive=True)
        #file_list = os.listdir(self.root_dir)
        #self.data_list = [x for x in file_list if x.endswith(".png")]
        #self.coco = COCO(self.anno_file)
        #with open(self.anno_file) as tmp:
        #    self.anno = json.load(tmp)
        # TODO

    def _load_image(self, index: int):
        img_path = self.data_list[index]
        image = Image.open(img_path)
        return image, img_path
        # return image

    def __getitem__(self, index: int):
        image, img_path = self._load_image(index)
        if self.transforms is not None:
            image = self.transforms(image=image)
        return image, img_path
        # return image and target

    def __len__(self) -> int:
        return len(self.data_list)
        # return the length of dataset
