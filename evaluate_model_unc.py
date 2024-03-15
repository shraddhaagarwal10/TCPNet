import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
# from torchmetrics import Dice
# from metrics import dice_coeff, dice_loss, bce_dice_loss, iou_metric
# from torchmetrics import Dice,JaccardIndex
# from metrics import bce_dice_loss, dice_coeff, iou_metric
from torchmetrics import Dice, JaccardIndex
import os

# SMOOTH = 1e-10
# def iou_score(outputs: torch.Tensor, labels: torch.Tensor):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.flatten(0, 1)  # BATCH x 1 x H x W => BATCH x H x W
#     labels = labels.flatten(0, 1)
#     intersection = (outputs & labels).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
#     return thresholded

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(3)

def evaluate(net, dataloader, device, amp, samples=1):
    # net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    
    # ce_loss=0
    # dice_loss=0
    iou = 0
    counts = 0
    val_loss = 0
    Iou = JaccardIndex(task="multiclass", num_classes=2).to(device)
    dice = Dice(num_classes=2).to(device)
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]
            mask_p = torch.empty(mask_true.shape[0],2,128,128).to(device=device)
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device).long()
            # predict the mask
            with torch.no_grad():
                # val_loss_s = 0
                # dice_score_s = 0
                # iou_s = 0
                for s in range(samples):
                # net = net.eval()
                    pred = net(image)
                    mask_p += pred
            mask_pred = mask_p/samples
            # print("********Pred_avg:", mask_pred)
            
            # val_loss = loss(logit=mask_pred, var=mask_var, true=mask_true.squeeze(1).long()) + nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1).long()) + (1-dice(mask_pred, mask_true.squeeze(1).long()))
            # ce_loss += nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1)) 
            # dice_loss += (1-dice(mask_pred, mask_true))
            
            # val_loss += nn.BCEWithLogitsLoss()(mask_pred, mask_true.float())
            # compute the Dice score, ignoring background
             # (reduce_batch_first=False)
            # jaccard = JaccardIndex(task="binary", num_classes=2).to(device=device)
            # iou = jaccard(mask_pred.argmax(1), mask_true.squeeze(1).long())
            # val_loss += val_loss_s
            # dice_score += dice_score_s
            # iou += iou_s
            val_loss += nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1)) + (1-dice(pred, mask_true))
            dice_score += dice(mask_pred, mask_true)
            iou += Iou(mask_pred, mask_true.squeeze(1))
            counts +=1
        #  max(num_val_batches, 1),          
    net.train()
    return val_loss/counts, dice_score/counts, iou/counts
