import torch
from tqdm import tqdm
import torch.nn as nn
from torchmetrics import Dice, JaccardIndex

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def evaluate(net, dataloader, device, amp, samples=1):
    num_val_batches = len(dataloader)
    iou = 0
    counts = 0
    val_loss = 0
    Iou = JaccardIndex(task="multiclass", num_classes=2).to(device)
    dice = Dice(num_classes=2).to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]
            mask_p = torch.zeros_like(torch.Tensor(mask_true.shape[0],2,128,128)).to(device=device)

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device).long()

            # predict the mask
            with torch.no_grad():
                for s in range(samples):
                    pred = net(image)
                    mask_p += pred
            mask_pred = mask_p/samples
            val_loss += nn.CrossEntropyLoss()(mask_pred, mask_true.squeeze(1)) + (1-dice(pred, mask_true))
            iou += Iou(mask_pred, mask_true.squeeze(1))
            counts +=1
      
    net.train()
    return val_loss/counts,  iou/counts
