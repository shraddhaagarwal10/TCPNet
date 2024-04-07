import argparse
import random, os
from dataset import BrainMRIDATASET
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from bayesian_crossentropy_loss import bayesian_categorical_crossentropy
from evaluate_data_model_unc import evaluate
from test_data_model_unc import test
from tcpnet import TCPNet
from torchmetrics import Dice
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

def train_model(
        model,
        device,
        data_tr,
        data_val,
        epochs: int = 200,
        learning_rate: float = 1e-5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        cp = 0,
        samples: int=20,
        ):

    optimizer = optim.Adam(model.parameters(),
                            lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    dice = Dice(num_classes=2).to(device)
    best_iou = 0
    for epoch in range(1, epochs + 1):
        print(f"Training for Epoch {epoch}, lambda {cp} and samples {samples}")
        model.train()
        epoch_loss = 0
        train_loss = []
        loss_val = []
        iou_val = []
        with tqdm(total=len(data_tr), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in data_tr:

                images, true_masks = batch[0], batch[1]

                assert images.shape[1] == 1, \
                    f'Network has been defined with {1} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device)
                
                with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
                    optimizer.zero_grad(set_to_none=True)
                    loss = 0
                    for s in range(samples):
                        pred, var = model(images)
                        loss1 = nn.CrossEntropyLoss()(pred, true_masks.squeeze(1).long()) + (1 - dice(pred, true_masks.long())) + cp*(bayesian_categorical_crossentropy()(logit=pred, var=var, true=true_masks.squeeze(1).long()))
                        grad_scaler.scale(loss1).backward()
                        loss += loss1/samples
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step
                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    val_loss, iou_score = evaluate(model, data_val, device, amp, cp)
                
                    train_loss.append(loss.cpu().detach().numpy())
                    loss_val.append(val_loss.cpu().detach().numpy())
                    iou_val.append(iou_score.cpu().detach().numpy())

        
        train_mean = np.mean(np.array(train_loss))
        loss_val_mean = np.mean(np.array(loss_val))
        iou_val_mean = np.mean(np.array(iou_val))

        # Save the model at the epch with best IOU Score       
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model, f'tcpnet_data_model_unc_{cp}_{samples}.pt')
        print("Train_loss:", train_mean )
        print("Validation_loss", loss_val_mean )
        print("IOU metric", iou_val_mean)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-path', '-d', dest='data_path', help='Path of data')
    parser.add_argument('--epochs', '-e', dest='epochs', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', '-l', dest='learning_rate', metavar='LR', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--sample', '-s', dest='sample', type=int, default=20, help='Number of forward passes')
    parser.add_argument('--lambda', '-lmd', dest='cp', type=float, default=0.01, help='Weight given to Bayesian crossentropy loss')
    parser.add_argument('--classes', '-c', dest='classes', type=int, default=2, help='Number of classes')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_ob = BrainMRIDATASET()
    dataset_tr, dataset_val, dataset_test  = data_ob.get_BrainMRIDataset(root=args.data_path)
    data_loader_tr = DataLoader(dataset_tr, batch_size=args.batch_size)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    print("***************** Training **********************")
    model = TCPNet(out_channels = args.classes)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    model.to(device=device)

    train_model(
        model=model,
        data_tr=data_loader_tr,
        data_val=data_loader_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        samples=args.sample,
        cp=args.cp,
        device=device,
    )

    model = torch.load(f'tcpnet_data_model_unc_{args.cp}_{args.sample}.pt')

    precision, recall, specificity, dice, iou, model_unc, data_unc = test(model,dataset_test,device,args.sample)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Specificity: ", specificity)
    print("Dice: ", dice)
    print("IOU: ", iou)
