from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
import torch.nn as nn
import numpy as np

class TripletSimSiamDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
        self.class_inds_diff = [torch.where(torch.tensor(self.raw_dataset.targets) == class_idx)[0] for class_idx in self.raw_dataset.class_to_idx.values()]
        self._set_class_inds()
        self.shuffled_idx = torch.randperm(len(self.raw_dataset))


    def __len__(self):
        return len(self.raw_dataset)
    
    def _set_class_inds(self):
        self.class_inds_equal = [torch.where(torch.tensor(self.raw_dataset.targets) == class_idx)[0] for class_idx in self.raw_dataset.class_to_idx.values()]

    def __getitem__(self, idx):
        img1 = self.raw_dataset[self.shuffled_idx[idx]]
        ind_img2 = np.random.choice(self.class_inds_equal[img1[1]])
        ind_img3 = np.random.choice(torch.cat(self.class_inds_diff[:img1[1]]+self.class_inds_diff[img1[1]+1:]))
        img2 = self.raw_dataset[ind_img2]
        img3 = self.raw_dataset[ind_img3]
        self.class_inds_equal[img1[1]] = self.class_inds_equal[img1[1]][self.class_inds_equal[img1[1]]!=ind_img2]
        if sum([len(x) for x in self.class_inds_equal])==0:
            self._set_class_inds()
            self.shuffled_idx = torch.randperm(len(self.raw_dataset))
        return img1, img2, img3

class AdvOutDistEvalDataset(Dataset):
    def __init__(self, raw_dataset, target, size=100):
        self.raw_dataset = raw_dataset
        self.size = size
        self.class_inds = [torch.where(torch.tensor(self.raw_dataset.targets) == class_idx)[0] for class_idx in self.raw_dataset.class_to_idx.values()]
        self.inds= np.random.choice(torch.cat(self.class_inds[:target]+self.class_inds[target+1:]), size=self.size)
        self.target = target
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.raw_dataset[self.inds[idx]]

class ClassDistDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, idx):
        return idx, self.raw_dataset[idx]


class View(nn.Module):
    """View layer
    
    Helper module for reshaping within torch.nn.Sequential.

    Args:
        dim: reshaping dimensions
    
    Input:
        x: torch.tensor
    
    Output:
        Reshaped torch.tensor
    """
    def __init__(self, *dim):
        super(View, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        x = x.view(self.dim)
        return x

def sample_from_latent(size_1, size_2):
    u = Uniform(-1, 1)
    return u.sample(size_1), u.sample(size_2) 


class OneHotEncode(nn.Module):
    def __init__(self, n_dims=None):
        super(OneHotEncode, self).__init__()
        self.n_dims = n_dims
    
    def forward(self, y):
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = self.n_dims if self.n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(*y.shape, -1)
        return Variable(y_one_hot).to(y.device) if isinstance(y, Variable) else y_one_hot.to(y.device)

def pgd(model, x_batch, target, k, eps, eps_step, kl_loss=False):
    if kl_loss:
        # loss function for the case that target is a distribution rather than a label (used for TRADES)
        loss_fn = torch.nn.KLDivLoss(reduction='sum')
    else:
        # standard PGD
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():  # disable gradients here
        # initialize with a random point inside the considered perturbation region
        x_adv = x_batch.detach() + eps * (2 * torch.rand_like(x_batch) - 1)
        #x_adv.clamp_(min=0.0, max=1.0)  # project back to the image domain

        for step in range(k):
            # make sure we don't have a previous compute graph and enable gradient computation
            x_adv.detach_().requires_grad_()

            with torch.enable_grad():  # re-enable gradients
                # run the model and obtain the loss
                out = F.log_softmax(model(x_adv)[0], dim=1) if kl_loss else model(x_adv)[0]
                model.zero_grad()
                # compute gradient
                loss_fn(out, target).backward()

            # compute step
            step = eps_step * x_adv.grad.sign()
            # project to eps ball
            x_adv = x_batch + (x_adv + step - x_batch).clamp(min=-eps, max=eps)
            # clamp back to image domain; in contrast to the previous exercise we clamp at each step (so this is part of the projection)
            # both implementations are valid; this dents to work slightly better
            #x_adv.clamp_(min=0.0, max=1.0)
    return x_adv.detach()


if __name__ == "__main__":
    pass
