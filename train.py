import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import logging
from pycocotools.coco import COCO
from model import DA_model
import torchvision
import argparse
import json
import torchvision.transforms as T
import matplotlib.pyplot as plt


num_workers = os.cpu_count()
n_classes = 9



parser = argparse.ArgumentParser()
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--data_root_dir', type=str,
                    default='./hw3_dataset')
parser.add_argument('--infer_root_dir', type=str,
                    default='./')
parser.add_argument('--batch_size', type=int, default=1)
#parser.add_argument('--train_horizon_flip_prob', type=float, default=0)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--output_path', type=str, default='./pred.json')
parser.add_argument('--lr_gamma', type=float, default=0.66)
parser.add_argument('--lr_dec_step_size', type=int, default=5)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--not_train', action='store_true')
parser.add_argument('--not_adaption', action='store_true')
parser.add_argument('--train1_dir', type=str, default='org/train')
parser.add_argument('--train2_dir', type=str, default='fog/train')
parser.add_argument('--val_dir', type=str, default='fog/val')
parser.add_argument('--checkpoint_index', type=int, default=4)


args = parser.parse_args()
# version = torch.version.__version__[:5]
# print('torch version is {}'.format(version))



def train_one_epoch(model, optimizer, source_loader, target_loader, epoch):
    model.train()
    print(min(len(source_loader), len(target_loader)))

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
    for i, ((source_images, source_labels), target_images) in enumerate(pbar):
        #print(target_images)

        source_images = list(image.to('cuda', non_blocking=True) for image in source_images) # list of [C, H, W]
        source_labels = [{k: v.to('cuda', non_blocking=True) for k, v in t.items()} for t in source_labels]
        target_images = list(image.to('cuda', non_blocking=True) for image in target_images) # list of [C, H, W]
        #target_images = list(target_images.to('cuda', non_blocking=True)) # list of [C, H, W]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = model(source_images, source_labels, target_images)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix({'loss': losses.item()})


@torch.no_grad()
def validation(model, data_loader):
    metric = MeanAveragePrecision()
    model.eval()

    for images, targets in tqdm(data_loader):
        images = list(image.to('cuda', non_blocking=True) for image in images)
        predictions = model(images)
        #predictions = ...  # postprocess: modify format to meet metric's requirements
        #targets = [target.to('cuda', non_blocking=True) for target in targets]
        targets = [{k: v.to('cuda', non_blocking=True) for k, v in target.items()} for target in targets]
        metric.update(predictions, targets)
    
    result = metric.compute()
    return result['map_50']


def build_dataloader():

    def collate_fn(batch):
        #print('hi')
        #if len(batch[0]) == 1:
        #    return tuple(*batch)
        #else:
        #    return tuple(zip(*batch))
        return tuple(zip(*batch))
        # TODO
        # depends on your code
        # return tuple(zip(*batch))

    # TODO
    from dataset import SourceDataset, TargetDataset, get_transform, TestDataset
    transform = get_transform(transType='valid')
    source_dataset = SourceDataset(args.data_root_dir, split=args.train1_dir, transform=transform)
    target_dataset = TargetDataset(args.data_root_dir, split=args.train2_dir, transform=transform)
    val_dataset = SourceDataset(args.data_root_dir, split=args.val_dir, transform=transform)
    test_dataset = TestDataset(args.infer_root_dir, split="", transform=transform)

    source_loader = torch.utils.data.DataLoader(source_dataset, args.batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)
    target_loader = torch.utils.data.DataLoader(target_dataset, args.batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, num_workers=num_workers, shuffle=False)

    return source_loader, target_loader, val_loader, test_loader


model = DA_model(n_classes, load_source_model=False, training=True, not_adaption=args.not_adaption)
model = model.to('cuda')


# TODO
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=args.lr_dec_step_size,
                                            gamma=args.lr_gamma)
num_epochs = args.num_epochs

map_list = []
best_epoch = 0
source_loader, target_loader, val_loader, test_loader = build_dataloader()
if not args.not_train:
    best_map = map_50 = validation(model, val_loader)
    map_list.append(map_50)
    print(map_50)
    torch.save(model.model.state_dict(), os.path.join(args.model_save_dir, 'model0.pt'))
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, source_loader, target_loader, epoch)
        scheduler.step()

        map_50 = validation(model, val_loader)
        map_list.append(map_50)
        print(map_50)
        torch.save(model.model.state_dict(), os.path.join(args.model_save_dir, 'model' + str(epoch+1) + '.pt'))
        #if map_50 > best_map:
        #    best_map = map_50
        #    best_epoch = epoch
        #    torch.save(model.model.state_dict(), args.model_save_path)
    epoch_list = [i for i in range(num_epochs+1)]
    plt.plot(epoch_list, map_list, color='blue', linestyle="-", linewidth="2", markersize="12", marker=".")
    plt.grid()
    plt.xlabel('Epoch', fontsize="10")
    plt.ylabel('mAP', fontsize="10")
    #plt.savefig('./mapPlot.png')



#output_list = []
if args.not_train:
    model = DA_model(n_classes, load_source_model=True, training=False, model_path=os.path.join(args.model_save_dir, 'model' + str(args.checkpoint_index) + '.pt'))
    model = model.to('cuda')

    #checkpoint = torch.load(args.model_save_path)
    #model.load_state_dict(checkpoint)
    model.eval()
    outputData = {}
    for images, img_path in tqdm(test_loader):
        #print(images, img_path)
        images = list(image.to('cuda', non_blocking=True) for image in images)
        imgPred = model(images)[0]
        imgPred['labels'] = imgPred['labels'].tolist()
        imgPred['boxes'] = imgPred['boxes'].tolist()
        imgPred['scores'] = imgPred['scores'].tolist()
        #uselessLen = img_path[0].find(args.data_root_dir)
        if args.infer_root_dir.endswith('/'):
            img_path = img_path[0][len(args.infer_root_dir):]
        else:
            img_path = img_path[0][len(args.infer_root_dir)+1:]
        outputData[img_path] = imgPred

#anno_file = os.path.join(args.data_root_dir, "fog/val.coco.json")
#with open(anno_file) as tmp:
#    anno = json.load(tmp)
#for i, img in enumerate(anno['images']):
#    outputData[img['file_name']] = best_output_list[i]
    with open(args.output_path, 'w+') as jsonfile:
        json.dump(outputData, jsonfile)
