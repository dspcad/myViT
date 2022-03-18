from nyuv2.nyu import NyuV2Dataset
import numpy as np
import torch, os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
import time

from torchvision.utils import save_image
import torch.distributed as dist
from torchmetrics import JaccardIndex
from vit import ViT

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()



def save_model(net, optimizer, epoch,save_path, distributed):
    if get_rank()==0:
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)



if __name__ == "__main__":

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #transform_train = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(64)])
    train_data_path = "/home/us000147/project/semantic_seg_and_depth/nyuv2/train/"
    nyu_v2_train = NyuV2Dataset(train_data_path+"images/", train_data_path+"label_depth/", train_data_path+"label_semantic_seg/",transform=transform_train)
    nyu_v2_train_sampler = torch.utils.data.RandomSampler(nyu_v2_train)


    val_data_path = "/home/us000147/project/semantic_seg_and_depth/nyuv2/val/"
    nyu_v2_val = NyuV2Dataset(val_data_path+"images/", val_data_path+"label_depth/", val_data_path+"label_semantic_seg/",transform=transform_train)



    batch_size = 2
    distributed = False
    save_path = "/home/us000147/tutorial/self_attention_cv/saved_model"
    nyu_v2_train_dataloader = DataLoader(nyu_v2_train, batch_size=batch_size, shuffle=False, sampler=nyu_v2_train_sampler, num_workers=8)
    nyu_v2_val_dataloader = DataLoader(nyu_v2_val, batch_size=8, shuffle=False, num_workers=8)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=41).to(device)
    #model = timm.models.vit_base_patch16_224_in21k(pretrained=True).to(device)
    #print(model.head.in_features)
    #model.head = torch.nn.Linear(model.head.in_features, 224*224*41).to(device)

    model = ViT(img_h=480, 
                img_w=640,
                in_channels=3,
                patch_dim=16,
                dim=10496,
                blocks=6,
                heads=16,
                dim_linear_block=10496,
                dim_head=64,
                dropout=0.0,
                num_classes=41,
                split_gpus=True)


    #model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=41).to(device)
    #print(model)

    

    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 1e-4
    #semantic_seg_optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    semantic_seg_optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0001)


    #semantic_seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(semantic_seg_optimizer, T_max=len(nyu_v2_train_dataloader))
    #semantic_seg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(semantic_seg_optimizer, T_max=len(nyu_v2_train_dataloader), eta_min=1e-6)
    semantic_seg_scheduler = torch.optim.lr_scheduler.MultiStepLR(semantic_seg_optimizer, milestones=[20,80,150], gamma=0.2)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Set loss function

    seg_ave_loss = 0.0
    for epoch in range(201):
        depth_data_available = True
        t_data_0 = time.time()

        nyu_v2_progress_bar = tqdm(nyu_v2_train_dataloader)
        nyu_v2_itr = iter(nyu_v2_progress_bar)
        b_seg_idx  = 0

        cnt=0
        while depth_data_available:
            ########################################
            #          Depth Estimation            #
            ########################################
            try:
                img, seg_target, depth_target = next(nyu_v2_itr)
                img = img.cuda(0)


                seg_target = seg_target.squeeze(dim=1)
                depth_target = depth_target.squeeze(dim=1)
                t_data_1 = time.time()
                img = img.cuda()
                seg_target = seg_target.cuda()
                global_step = epoch * len(nyu_v2_train_dataloader) + b_seg_idx
                                
                #target = target.reshape(batch_size,h,w)
                #save_image(img, f"test_{cnt}.png")

                t_net_0 = time.time()
                #pred = model(img)['out']
                pred = model(img)
                seg_target = seg_target.to(pred.device)

                #pred = pred.reshape(-1,41,224,224)
                #pred = pred.reshape(-1,41,64,64)



                #print(type(outputs))
                #print(outputs['out'].shape)

                #print(f"label: {seg_target.shape}")
                #print(f"pred: {pred.shape}")
                seg_loss = criterion(pred,seg_target.long())
                #print(seg_loss.item())
                semantic_seg_optimizer.zero_grad()
                seg_loss.backward()
                semantic_seg_optimizer.step()
                #semantic_seg_scheduler.step()
                t_net_1 = time.time()


                seg_ave_loss = (seg_ave_loss*global_step + float(seg_loss) )/(global_step+1)



                if hasattr(nyu_v2_progress_bar,'set_postfix'):
                    nyu_v2_progress_bar.set_postfix(loss = '%.3f' % float(seg_ave_loss), lr= '%.8f' % float(semantic_seg_scheduler.get_last_lr()[0]), cur_loss= '%.3f' % float(seg_loss),
                                            data_time = '%.3f' % float(t_data_1 - t_data_0),
                                            net_time = '%.3f' % float(t_net_1 - t_net_0))



                t_data_0 = time.time()
                b_seg_idx += 1
                
                #if cnt==5:
                #    break
 
                #cnt += 1    

            except StopIteration:
                depth_data_available = False  # 
                print(f"Epoch {epoch} is Done.")
                del nyu_v2_itr
                break
        
        semantic_seg_scheduler.step()
        if epoch%50==0 and epoch > 0:
            save_model(model, semantic_seg_optimizer, epoch,save_path, distributed)

            #model.eval()
            #eval_res = JaccardIndex(num_classes=41, ignore_index=0).cuda()
            #with torch.no_grad():
            #    for img, seg_target, depth_target in tqdm(nyu_v2_val_dataloader):

            #        seg_target = seg_target.squeeze(dim=1)
            #        depth_target = depth_target.squeeze(dim=1)
            #        img = img.cuda()
            #        seg_target = seg_target.cuda()
            #        #pred = model(img)['out']

            #        pred = model(img)
            #        #pred = pred.reshape(-1,41,64,64)
            #        #print(f"pred: {pred.shape}")
            #        #print(f"target: {seg_target.shape}")
            #        #print(f"target: {target.shape}   {target[0][i][j]}")
            #        eval_res(pred,seg_target)
            #        
            #    print(eval_res.compute())

            #model.train()

