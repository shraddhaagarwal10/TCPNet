import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BrainMRIDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks ):
        self.imgs = imgs 
        self.masks = masks

    def __getitem__(self,indx):
        img = self.imgs[indx]
        mask = self.masks[indx]
        return img, mask

    def __len__(self):
        return len(self.imgs)
    
class BrainMRIDATASET:
    def get_BrainMRIDataset(self, root):
        imgs = np.clip((np.load(os.path.join(root, 'images.npy'))/12728),0,1)
        mask = np.load(os.path.join(root, 'masks.npy'))
        pid = np.load(os.path.join(root, 'pids.npy'))

        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = self.prepare_data(imgs=imgs, mask=mask, pid=pid)
        
        # augument with Horizontal Flip
        aug_imgs_tr = torch.cat((X_train, transforms.RandomHorizontalFlip(p = 1)(X_train)),0)
        aug_masks_tr = torch.cat((Y_train, transforms.RandomHorizontalFlip(p = 1)(Y_train)),0)

        train_ds = BrainMRIDataset(aug_imgs_tr,aug_masks_tr)
        val_ds = BrainMRIDataset(X_valid, Y_valid)
        test_ds = BrainMRIDataset(X_test, Y_test)
        return train_ds, val_ds, test_ds

    # split the data to train, validation and test sets
    def prepare_data(self, imgs, mask,pid):
        img_dict = {}
        for i in range(len(pid)):
            p = pid[i][0][0]
            if p in img_dict:
                img_dict[p].append(i)
            else:
                img_dict[p] = [i]

        keys = list(img_dict.keys())
        train_keys = keys[:143]
        val_keys = keys[143:143+45]
        test_keys = keys[188:]

        print(f"Number of patients in train set: {len(train_keys)}, validation_set: {len(val_keys)} and test set: {len(test_keys)}")

        train_images = []
        train_mask = []
        for p_key in train_keys:
            img_list = img_dict[p_key]
            for img_idx in img_list:
                train_images.append(imgs[img_idx])
                train_mask.append(mask[img_idx])
        train_images = np.array(train_images)
        train_mask = np.array(train_mask)
        train_images, train_mask = self.transform_to_tensor(train_images, train_mask)

        val_images = []
        val_mask = []
        for p_key in val_keys:
            img_list = img_dict[p_key]
            for img_idx in img_list:
                val_images.append(imgs[img_idx])
                val_mask.append(mask[img_idx])
        val_images = np.array(val_images)
        val_mask = np.array(val_mask)
        val_images, val_mask = self.transform_to_tensor(val_images, val_mask)

        test_images = []
        test_mask = []
        for p_key in test_keys:
            img_list = img_dict[p_key]
            for img_idx in img_list:
                test_images.append(imgs[img_idx])
                test_mask.append(mask[img_idx])
        test_images = np.array(test_images)
        test_mask = np.array(test_mask)
        test_images, test_mask = self.transform_to_tensor(test_images, test_mask)
        return train_images, train_mask, val_images, val_mask, test_images, test_mask

    # convert to torch tensor and transform the resolution of image to 128x128
    def transform_to_tensor(self,img, mask):
        images = torch.Tensor(img)
        masks = torch.Tensor(mask)
        images = torch.unsqueeze(images, axis=1)
        masks = torch.unsqueeze(masks, axis=1)
        transform_img = transforms.Resize(128)
        transform_mask = transforms.Resize(128, interpolation=InterpolationMode.NEAREST)
        final_images = transform_img(images)
        final_masks = transform_mask(masks)
        return final_images, final_masks
    
