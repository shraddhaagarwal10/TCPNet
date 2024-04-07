import torch
from tqdm import tqdm
import torch.nn as nn
from torchmetrics import Dice,JaccardIndex
from bayesian_crossentropy_loss import bayesian_categorical_crossentropy
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def evaluate(net, dataloader, device, amp, cp=0, samples=1):
    num_val_batches = len(dataloader)
    iou = 0
    counts = 0
    val_loss = 0
    Iou = JaccardIndex(task="multiclass", num_classes=2).to(device)
    dice = Dice(task="multiclass", num_classes=2).to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device).long()

            mask_p = torch.zeros_like(torch.Tensor(mask_true.shape[0],2,128,128)).to(device=device)
            var_p = torch.zeros_like(torch.Tensor(mask_true.shape[0],1,128,128)).to(device=device)

            # predict the mask
            with torch.no_grad():
                for s in range(samples):
                    pred, var = net(image)
                    mask_p += pred
                    var_p += var
                mask_pred = mask_p/samples
                mask_var = var_p/samples

            print(mask_var)
            try:
                val_loss += nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1).long()) + (1-dice(mask_pred, mask_true)) + cp*(bayesian_categorical_crossentropy()(logit=mask_pred, var=mask_var, true=mask_true.squeeze(1)))
            except:
                pass

            iou += Iou(mask_pred, mask_true.squeeze(1))
            counts +=1
        #  max(num_val_batches, 1),          
    net.train()
    return val_loss/counts, iou/counts
