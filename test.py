from nyuv2.nyu import NyuV2Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
import time

from torchvision.utils import save_image

from vit import ViT
from torchmetrics import JaccardIndex
import argparse, os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the FCN semantic segmentation model with NYU depth V2')
    parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
    parser.add_argument('--data_path', type=str, default='/home/us000147/project/semantic_seg_and_depth/nyuv2/val/', help="path to the data folder")
    parser.add_argument('--dataset', type=int, default=0, help='0: semantic segmentation, 1: depth estimation')
    parser.add_argument('--model_path', type=str, default='/home/us000147/tutorial/self_attention_cv/saved_model/ep000.pth', help="path to the saved model folder")
    
    opt = parser.parse_args()
    print(opt)


    opt.val_data_path               = opt.data_path + 'images/'
    opt.val_label_semantic_seg_path = opt.data_path + 'label_semantic_seg/'
    opt.val_label_depth_path        = opt.data_path + 'label_depth/'


    #transform_train = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    nyu_v2_val = NyuV2Dataset(opt.val_data_path, opt.val_label_depth_path, opt.val_label_semantic_seg_path,transform=transform_train)
    nyu_v2_val_dataloader = DataLoader(nyu_v2_val, batch_size=8, shuffle=False, num_workers=8)

    distributed = False



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViT(img_h=480, 
                 img_w=640,
                 in_channels=3,
                 patch_dim=8,
                 dim=2624,
                 blocks=6,
                 heads=24,
                 dim_linear_block=64,
                 dim_head=64,
                 dropout=0,
                 num_classes=41).to(device)

    model.load_state_dict(torch.load(opt.model_path)['model'])

    
    model.eval()
    eval_res = JaccardIndex(num_classes=41, ignore_index=0, compute_on_step=False).cuda()
    with torch.no_grad():
        for img, seg_target, depth_target in tqdm(nyu_v2_val_dataloader):
            seg_target = seg_target.squeeze(dim=1)
            depth_target = depth_target.squeeze(dim=1)
            img = img.cuda()
            seg_target = seg_target.cuda()
            pred = model(img)

            
            #print(f"pred: {pred.shape}  {type(pred)}")
            #print(f"    {pred[0,:,0,0]}")
            #print(f"target: {target.shape}    {type(target)}")
            #print(f"    {target[0][0][0]}")
            eval_res(pred,seg_target)
            
        print(eval_res.compute())

