import argparse
import logging
import os
import random
import sys
# from sklearn.model_selection import train_test_split
from dataset import BrainMRIDATASET
import nibabel
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
from evaluate_model_unc import evaluate
from unet_dropout import UNet
from torchmetrics import Dice
# from utils.data_loading import BasicDataset, CarvanaDataset
# from metrics import bce_dice_loss

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
    
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
        epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
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

    optimizer = optim.Adam(model.parameters(),
                            lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # nn.CrossEntropyLoss() if model.n_classes > 1 else 
    global_step = 0
    dice = Dice(num_classes=2).to(device)

    # print("Length", len(data_tr))
    train_loss_ep = []
    ce_loss_val_ep = []
    dice_loss_val_ep = []
    loss_val_ep = []
    dice_val_ep = []
    iou_val_ep = []
    best_iou = 0
    for epoch in range(1, epochs + 1):
        print(f"Training for Epoch {epoch} and Sample {samples}")
        model.train()
        epoch_loss = 0
        train_loss = []
        ce_loss_val = []
        dice_loss_val = []
        loss_val = []
        dice_val = []
        iou_val = []
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
                # print(images.shape)
                true_masks = true_masks.to(device=device)
                # print(true_masks.shape)
                # true_masks = torch.unsqueeze(true_masks, dim = 1)
                # print(true_masks.shape)

                with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    optimizer.zero_grad(set_to_none=True)
                    loss = 0
                    for s in range(samples):
                        pred1 = model(images)
                        loss1 = nn.CrossEntropyLoss()(pred1, true_masks.squeeze(1).long()) + (1 - dice(pred1, true_masks.long()))
                        grad_scaler.scale(loss1).backward()
                        loss += loss1/samples



                    # pred2 = model(images)
                    # loss2 = nn.CrossEntropyLoss()(pred2, true_masks.squeeze(1).long()) + (1 - dice(pred2, true_masks.long()))
                    # grad_scaler.scale(loss2).backward()

                    # pred3 = model(images)
                    # loss3 = nn.CrossEntropyLoss()(pred3, true_masks.squeeze(1).long()) + (1 - dice(pred3, true_masks.long()))
                    # grad_scaler.scale(loss3).backward()

                    # pred4 = model(images)
                    # loss4 = nn.CrossEntropyLoss()(pred4, true_masks.squeeze(1).long()) + (1 - dice(pred4, true_masks.long()))
                    # grad_scaler.scale(loss4).backward()

                    # pred5 = model(images)
                    # loss5 = nn.CrossEntropyLoss()(pred5, true_masks.squeeze(1).long()) + (1 - dice(pred5, true_masks.long()))
                    # grad_scaler.scale(loss5).backward()

                    # pred6 = model(images)
                    # loss6 = nn.CrossEntropyLoss()(pred6, true_masks.squeeze(1).long()) + (1 - dice(pred6, true_masks.long()))
                    # grad_scaler.scale(loss6).backward()

                    # pred7 = model(images)
                    # loss7 = nn.CrossEntropyLoss()(pred7, true_masks.squeeze(1).long()) + (1 - dice(pred7, true_masks.long()))
                    # grad_scaler.scale(loss7).backward()
                    # # print("Loss1:", loss1)
                    # print("Loss2:", loss2)
                    # print("Loss3:", loss3)
                    # print("Loss4:", loss4)
                    # print("Loss5:", loss5)
                    # loss = (loss1 + loss2 + loss3 )/3
                    # print("Loss:", loss)
                    # print(masks_pred.shape, true_masks.shape)
                    # loss = bce_dice_loss(masks_pred.argmax(1).float(), true_masks.squeeze(1).float())
                    # print(loss) #+ torch.mean(0.5*torch.log(mask_var + 1e-10))
                    # loss = nn.BCEWithLogitsLoss()(masks_pred, true_masks.float())
                    # mask_pred_dl = F.softmax(masks_pred, dim=1).float()
                    # print(mask_pred_dl.shape)
                    # true_mask_dl = F.one_hot(true_masks).permute(0, 3, 1, 2).float()
                    # # print(mask_pred_dl.shape, true_mask_dl.shape)
                    # loss += dice_loss(mask_pred_dl, true_mask_dl, multiclass=False)
                    # loss += torch.mean(0.5*torch.log(variance))
                    # grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step
                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    # experiment.log
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    val_loss, dice_score, iou_score = evaluate(model, data_val, device, amp, samples)
                    # scheduler.step(val_loss)
                    # scheduler.step(dice_score)
                    # scheduler.step(iou_score)

                    # logging.info('Validation Loss: {}'.format(val_loss))
                    # logging.info('Validation Dice score: {}'.format(dice_score))
                    # logging.info('Validation IOU metric: {}'.format(iou_score))
                    
                    train_loss.append(loss.cpu().detach().numpy())
                    loss_val.append(val_loss.cpu().detach().numpy())
                    dice_val.append(dice_score.cpu().detach().numpy())
                    iou_val.append(iou_score.cpu().detach().numpy())

        
        train_mean = np.mean(np.array(train_loss))
        train_loss_ep.append(train_mean)

        loss_val_mean = np.mean(np.array(loss_val))
        loss_val_ep.append(loss_val_mean)

        dice_val_mean = np.mean(np.array(dice_val))
        dice_val_ep.append(dice_val_mean)

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
        # torch.save(model.state_dict(), '/DATA/shraddha/BS_Thesis/torch_implementation/model_hist_dataunc_500.pt')
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model, 'results/unet_model_unc_samp' + str(samples) + '.pt')
        print("Train_loss:", train_mean )
        print("Validation_loss", loss_val_mean )
        print("Dice coefficient", dice_val_mean)
        print("IOU metric", iou_val_mean)

        np.savetxt("results/train_loss_model_unc_samp" + str(samples) + ".csv", np.array(train_loss_ep), delimiter=",", fmt="%f")
        np.savetxt("results/val_loss_model_unc_samp" + str(samples) + ".csv", np.array(loss_val_ep), delimiter=",", fmt="%f")
        np.savetxt("results/val_dice_model_unc_samp" + str(samples) + ".csv", np.array(dice_val_ep), delimiter=",", fmt="%f")
        np.savetxt("results/val_iou_model_unc_samp" + str(samples) + ".csv", np.array(iou_val_ep), delimiter=",", fmt="%f")

    print(f"Model has trained for sample {samples}")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-path', '-d', dest='data_path', default='/DATA/shraddha/brain-tumor-segmentation-unet/brain_tumor_dataset/')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    data_ob = BrainMRIDATASET()
    dataset_tr, dataset_val, dataset_test  = data_ob.get_BrainMRIDataset(root=args.data_path)
    data_loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    print("***************** Training **********************")
    model = UNet(out_channels=2)

    model.to(device=device)
    # try:
    sample = 20
    # for sample in sample_list:
    train_model(
        model=model,
        data_tr=data_loader_tr,
        data_val=data_loader_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-4,#args.lr,
        device=device,
        samples=sample,
    )
