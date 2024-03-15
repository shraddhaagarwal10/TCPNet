import argparse
import logging
import os
import random
import sys
# from sklearn.model_selection import train_test_split
from dataset import BrainMRIDATASET
# import nibabel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from evaluate_data_model import evaluate
from unet_data_unc import UNet
# from loss import dice_coeff, dice_loss
# from torchmetrics import Dice
from torchmetrics import Dice


class bayesian_categorical_crossentropy(nn.Module):
    def __init__(self):
        super(bayesian_categorical_crossentropy, self).__init__()
        self.T = 30
        self.ELU = nn.ELU()
        self.num_classes = 2
        self.categorical_crossentropy = nn.CrossEntropyLoss()

    def bayesian_categorical_crossentropy_internal(self, logit, var, true):

        std = torch.sqrt(var) + 1e-15
        variance_depressor = torch.exp(var) - torch.ones_like(var)
        undistorted_loss = self.categorical_crossentropy(logit+1e-15,true) #In pytorch loss (output,target)

        #iterable = torch.autograd.Variable(np.ones(self.T))
        dist = torch.distributions.normal.Normal(torch.zeros_like(std), std)

        monte_carlo = [self.gaussian_categorical_crossentropy(logit, true, dist, undistorted_loss, self.num_classes) for _ in range(self.T)]
        monte_carlo = torch.stack(monte_carlo)
        variance_loss = torch.mean(monte_carlo,axis = 0) * undistorted_loss

        loss_final = variance_loss + undistorted_loss + variance_depressor
        # reduction of loss required. Taking mean() as that is what happens in batched crossentropy
        return loss_final.mean()

    def gaussian_categorical_crossentropy(self, logit, true, dist, undistorted_loss, num_classes):
        std_samples = torch.squeeze(torch.transpose(dist.sample((num_classes,)), 0,1))
        #print("########### pred",pred.shape," std samples ",std_samples.shape)
        distorted_loss = self.categorical_crossentropy(logit + 1e-15 + std_samples, true)
        diff = undistorted_loss - distorted_loss
        return -1*self.ELU(diff)

    def forward(self, logit, var, true):
        return self.bayesian_categorical_crossentropy_internal(logit, var, true)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(3)
# def get_transform(data, axis=1):
#     images = np.expand_dims(data, axis)
#     masks = np.expand_dims(mask, axis)
#     images = torch.Tensor(images)
#     masks = torch.Tensor(masks)
#     transform = transforms.Resize(128)
#     images = transform(images)
#     masks = transform(masks)
#     return images, masks

def train_model(
        model,
        device,
        data_tr,
        data_val,
        epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        loss_fn=None,
        cp = 0,
        samples: int=1,
        ):

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(data_tr)}
        Validation size: {len(data_val)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # optimizer = optim.Adam(model.parameters(),
    #                         lr=learning_rate, weight_decay=weight_decay, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # # nn.CrossEntropyLoss() if model.n_classes > 1 else 
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: maximize Dice score
    global_step = 0
    dice = Dice(num_classes=2).to(device)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # print("Length", len(data_tr))
    train_loss_ep = []
    loss_val_ep = []
    dice_val_ep = []
    iou_val_ep = []
    f1_val_ep = []
    best_iou = 0
    for epoch in range(1, epochs + 1):
        print(f"Training for Epoch {epoch}, lambda {cp} and samples {samples}")
        model.train()
        epoch_loss = 0
        train_loss = []
        loss_val = []
        dice_val = []
        iou_val = []
        f1_val = []
        with tqdm(total=len(data_tr), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in data_tr:
                # print(batch)
                images, true_masks = batch[0], batch[1]

                assert images.shape[1] == 1, \
                    f'Network has been defined with {1} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # print(images.shape)
                true_masks = true_masks.to(device=device)
                # true_masks = torch.unsqueeze(true_masks, dim = 1)
                # print(true_masks.shape)
                
                with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    optimizer.zero_grad(set_to_none=True)
                    loss = 0
                    for s in range(samples):
                        pred, var = model(images)
                        loss1 = nn.CrossEntropyLoss()(pred, true_masks.squeeze(1).long()) + (1 - dice(pred, true_masks.long())) + cp*(bayesian_categorical_crossentropy()(logit=pred, var=var, true=true_masks.squeeze(1).long()))
                        grad_scaler.scale(loss1).backward()
                        loss += loss1/samples
                    # pred, var = model(images)
                    # print(pred1.shape, true_masks.squeeze(1).long().shape)
                    # loss = criterion(pred, true_masks.squeeze(1).long()) + (1 - dice(pred, true_masks.long())) + cp*(bayesian_categorical_crossentropy()(logit=pred, var=var, true=true_masks.squeeze(1).long()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    # grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step
                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                
                    # # print(type(loss1),type(loss2),type(loss3))
                    # loss = torch.add(torch.add(loss1, loss2), loss3)/3
                    # if model.n_classes == 1:
                    # print(masks_pred.shape, true_masks.shape)
                    # print(masks_pred.shape, true_masks.shape)
                    # loss = criterion(masks_pred, true_masks.float())
                    # print(masks_pred.shape, true_masks.shape)
                    # dice = Dice().to(device=device)
                    # loss = loss_fn(logit=masks_pred, var=mask_var, true=true_masks.squeeze(1).long()) + nn.CrossEntropyLoss()(masks_pred, true_masks.squeeze(1).long()) + (1 - dice(masks_pred, true_masks.squeeze(1).long()))
                 #   loss = loss_fn(logit=masks_pred.squeeze(1), var=mask_var, true=true_masks.squeeze(1)) + bce_dice_loss(masks_pred, true_masks)
                    # scheduler.step(loss)
                    # mask_pred_dl = F.softmax(masks_pred, dim=1).float()
                    # print(mask_pred_dl.shape)
                    # true_mask_dl = F.one_hot(true_masks).permute(0, 3, 1, 2).float()s
                    # # print(mask_pred_dl.shape, true_mask_dl.shape)
                    # loss += dice_loss(mask_pred_dl, true_mask_dl, multiclass=False)
                    # loss += torch.mean(0.5*torch.log(variance))
                    # optimizer.zero_grad(set_to_none=True)
                    # grad_scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    # grad_scaler.step(optimizer)
                    # grad_scaler.update()
                  
                    # experiment.log({
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    val_loss, dice_score, iou_score = evaluate(model, data_val, device, amp, cp, samples)
                    
                    # scheduler.step(val_loss)
                    # scheduler.step(dice_score)
                    # scheduler.step(iou_score)

                    # logging.info('Validation Loss: {}'.format(val_loss))
                    # logging.info('Validation Dice score: {}'.format(dice_score))
                    # logging.info('Validation IOU metric: {}'.format(iou_score))
                    
                    train_loss.append(loss.cpu().detach().numpy())
                    loss_val.append(val_loss.cpu().detach().numpy())
                    # var_val.append(val_var.cpu().detach().numpy())
                    dice_val.append(dice_score.cpu().detach().numpy())
                    # f1_val.append(f1_score.cpu().detach().numpy())
                    iou_val.append(iou_score.cpu().detach().numpy())

        
        train_mean = np.mean(np.array(train_loss))
        train_loss_ep.append(train_mean)

        loss_val_mean = np.mean(np.array(loss_val))
        loss_val_ep.append(loss_val_mean)

        # var_val_mean = np.mean(np.array(var_val))
        # var_val_ep.append(var_val_mean)

        dice_val_mean = np.mean(np.array(dice_val))
        dice_val_ep.append(dice_val_mean)

        # f1_val_mean = np.mean(np.array(f1_val))
        # f1_val_ep.append(f1_val_mean)

        iou_val_mean = np.mean(np.array(iou_val))
        iou_val_ep.append(iou_val_mean)

        

    
    
                    # print("Train_loss:", loss.cpu().detach().numpy())
                    # print("Validation_loss", val_loss.cpu().detach().numpy())
                    # print("Dice coefficient", dice_score.cpu().detach().numpy())
                    # print("IOU metric", iou_score.cpu().detach().numpy())
        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     state_dict['mask_values'] = (0,1)
        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
        
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model, f'data_model_results/unet_data_model_unc_{cp}_{samples}.pt')
        print("Train_loss:", train_mean )
        print("Validation_loss", loss_val_mean )
        print("Dice coefficient", dice_val_mean)
        # print("F1-Score", f1_val_mean)
        print("IOU metric", iou_val_mean)

        np.savetxt(f"data_model_results/train_loss_data_model_unc_{cp}_{samples}.csv", np.array(train_loss_ep), delimiter=",", fmt="%f")
        np.savetxt(f"data_model_results/val_loss_data_model_unc_{cp}_{samples}.csv", np.array(loss_val_ep), delimiter=",", fmt="%f")
        np.savetxt(f"data_model_results/val_dice_data_model_unc_{cp}_{samples}.csv", np.array(dice_val_ep), delimiter=",", fmt="%f")
        # np.savetxt(f"data_model_results/val_f1_data_model_unc_{cp}_{samples}.csv", np.array(f1_val_ep), delimiter=",", fmt="%f")
        np.savetxt(f"data_model_results/val_iou_data_model_unc_{cp}_{samples}.csv", np.array(iou_val_ep), delimiter=",", fmt="%f")
    # np.savetxt("val_var_modelunc_mulloss.csv", np.array(var_val_ep), delimiter=",", fmt="%f")

    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-path', '-d', dest='data_path', default='/DATA/shraddha/brain-tumor-segmentation-unet/brain_tumor_dataset/')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # torch.manual_seed_all(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(3)
    CUDA_LAUNCH_BLOCKING=1
    args = get_args()

    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    data_ob = BrainMRIDATASET()
    dataset_tr, dataset_val, dataset_test  = data_ob.get_BrainMRIDataset(root=args.data_path)
    data_loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    print("***************** Training **********************")
    model = UNet(out_channels = args.classes)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    model.to(device=device)
    # try:
    # lamb = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    lamb = 0.01
    sample = 9
    # for l in lamb:
    train_model(
        model=model,
        data_tr=data_loader_tr,
        data_val=data_loader_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-5,
        device=device,
        cp = lamb,
        samples=sample,
    )
