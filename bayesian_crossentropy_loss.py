import torch
import torch.nn as nn

class bayesian_categorical_crossentropy(nn.Module):
    def __init__(self, T = 30, num_classes = 2):
        super(bayesian_categorical_crossentropy, self).__init__()
        self.T = T
        self.ELU = nn.ELU()
        self.num_classes = num_classes
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