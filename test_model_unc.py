import torch
from tqdm import tqdm
from torchmetrics import JaccardIndex, Precision, Recall , Specificity, F1Score

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def test(model, dataloader, device, samples):
        precision = Precision(task="multiclass", average='macro', num_classes=2).to(device)
        recall = Recall(task="multiclass", average='macro', num_classes=2).to(device)
        specificity = Specificity(task="multiclass", average='macro', num_classes=2).to(device)
        f1 = F1Score(task="multiclass", average='macro', num_classes=2).to(device)
        iou = JaccardIndex(task="multiclass", num_classes=2).to(device)
        prec = 0
        rec = 0
        spec = 0
        dice_sc = 0
        iou_sc = 0
        counts = 0

        # initialize metrcies to save model uncertainty values 
        model_unc= torch.empty(len(dataloader),2,128,128).to(device=device)

        with torch.autocast(device.type if device.type != 'cuda' else 'cpu'):
            for batch in tqdm(dataloader, total=len(dataloader)):
                image, mask_true = batch[0], batch[1]
                image = torch.unsqueeze(image, dim=0)
                mask_true = torch.unsqueeze(mask_true, dim=0)            
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)              
                mask_true = mask_true.to(device=device, dtype=torch.long)     

                mask_p = torch.zeros_like(torch.Tensor(mask_true.shape[0],2,128,128)).to(device=device)
                predict = torch.empty(samples,mask_true.shape[0],2,128,128).to(device=device)
                
                with torch.no_grad():
                    for s in range(samples):
                        mask_pred = model(image)
                        mask_p += mask_pred
                        predict[s] = mask_pred
                        
                mask_pred = mask_p/samples
                var_pred = torch.var(predict, dim=0)
                model_var, var_ind = torch.max(var_pred, axis = 1)
                model_unc[counts] = model_var

                prec += precision(mask_pred, mask_true.squeeze(1).long())
                rec += recall(mask_pred, mask_true.squeeze(1).long())
                spec += specificity(mask_pred, mask_true.squeeze(1).long())
                dice_sc += f1(mask_pred, mask_true.squeeze(1).long())
                iou_sc += iou(mask_pred, mask_true.squeeze(1).long())
                counts +=1

        prec_score = prec/counts
        rec_score = rec/counts
        spec_score = spec/counts
        dice_score = dice_sc/counts
        iou_score = iou_sc/counts
        return prec_score, rec_score,spec_score,dice_score, iou_score, model_unc