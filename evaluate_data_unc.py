import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
# from torchmetrics import Dice
# from metrics import dice_coeff, dice_loss, bce_dice_loss, iou_metric
from torchmetrics import Dice,JaccardIndex
import os
torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.manual_seed_all(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(3)

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

def evaluate(net, dataloader, device, amp, cp=0.1):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(3)
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou = 0
    counts = 0
    val_loss = 0
    Iou = JaccardIndex(task="multiclass", num_classes=2).to(device)
    dice = Dice(num_classes=2).to(device)
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device).long()
            # predict the mask
            with torch.no_grad():
                net = net.eval()
                # val_loss_s = 0
                # dice_score_s = 0
                # iou_s = 0
                mask_pred, mask_var = net(image)

            val_loss += nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1).long()) + (1-dice(mask_pred, mask_true)) + cp*(bayesian_categorical_crossentropy()(logit=mask_pred, var=mask_var, true=mask_true.squeeze(1).long()))
            # print(mask_pred.shape)
            # print(mask_true.shape)
            # print(mask_var.shape)
            # print(mask_true.shape, mask_pred.shape)
            # torch.nn.CrossEntropyLoss() if net.n_classes > 1 else
            # criterion = torch.nn.BCEWithLogitsLoss()
            # val_loss = criterion(mask_pred, mask_true.float())
            # metrics
            # assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # print(mask_true.shape, mask_pred.shape)
            # val_loss += dice_loss(mask_pred, mask_true.float(), multiclass=False)
            # dice = Dice().to(device=device)
          #  val_loss += 0.3*(loss(logit=mask_pred, var=mask_var, true=mask_true.squeeze(1).long())) + 0.7*(nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1).long())) # + (1-dice(mask_pred, mask_true.squeeze(1).long())))
            # val_loss += 0.9*(nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1))) + 0.1*torch.mean(torch.log(mask_var + 1 + 1e-10))
            # compute the Dice score, ignoring background
            # print(mask_pred)
            dice_score += dice(mask_pred, mask_true) # (reduce_batch_first=False)
            # jaccard = JaccardIndex(task="binary", num_classes=2).to(device=device)
            # iou = jaccard(mask_pred.argmax(1), mask_true.squeeze(1).long())
            iou += Iou(mask_pred, mask_true.squeeze(1))
            counts +=1
        #  max(num_val_batches, 1),          
    net.train()
    return val_loss/counts, dice_score/counts, iou/counts
