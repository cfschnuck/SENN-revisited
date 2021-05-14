import torch
import logging
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import torch.nn.functional as F

from .utils import AdvOutDistEvalDataset, ClassDistDataset

class LipschitzEvaluator():
    """Evaluator to calculate Lipschitz constant es defined in https://arxiv.org/abs/1806.07538

    Input:
        model: trained model
        eps: epsilon ball
        tol: improvement tolerance
        patience: patience periods
        max_iter: maximum iterations
        lr: learning rate
    """
    def __init__(self, model, eps = 0.01, tol = 1e-3, max_iter = 1e4, patience = 3, lr = 1e-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.eps = eps
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.patience = patience
    
    
    def local_lipschitz_estimate(self, x):
        x = Variable(x, requires_grad = False).to(self.device)
        if self.eps is not None:
            noise = (self.eps * torch.randn(x.size())).to(self.device)
            z = Variable(x.clone() + noise, requires_grad = True).to(self.device)
        else:
            z = Variable(torch.randn(x.size()), requires_grad = True).to(self.device)
        
        optim = torch.optim.Adam([z], lr=self.lr)

        hx, fx = self.model(x)[1]
        fx, hx = fx.detach(), hx.detach()
        
        i = 0
        best_loss = 1.e6
        no_improvement_count = 0
        while i < self.max_iter and no_improvement_count <= self.patience:
            i += 1
            optim.zero_grad()
            hz, fz  = self.model(z)[1]
            dist_f = (fz-fx).norm()
            dist_h = (z-x).norm()  
            loss = dist_h/dist_f
            lip = 1/loss.item()

            ## side constraint via lagrange
            if self.eps is not None:
                loss += 0.1 * F.relu(dist_h-self.eps) 
            
            loss.backward()
            optim.step()
            best_loss = min(best_loss, loss+self.tol)

            if i > 10 and best_loss < loss+self.tol:
                no_improvement_count += 1
        if i >= self.max_iter:
            logging.info("Convergence Warning: Lipschitz not converged")
        return lip, z
    
    def dataset_lipschitz_estimate(self, dataset, max_iter=None):
        self._freeze_weights()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        lips = AverageValueMeter()
        pbar = tqdm(data_loader, desc= "Lipschitz: ")
        for i, (x, _) in enumerate(pbar):
            x = Variable(x).to(self.device)
            lip, _ = self.local_lipschitz_estimate(x)
            lips.add(lip)
            if max_iter is not None and i == max_iter:
                break
            pbar.set_postfix({"Mean lipschitz ratio": lips.mean})
        self._unfreeze_weights()
        return lips.mean

    def _freeze_weights(self):
        for params in self.model.parameters():
            params.requires_grad = False

    def _unfreeze_weights(self):
        for params in self.model.parameters():
            params.requires_grad = True

class DistanceEvaluator():
    """Distance evaluator as defined in https://arxiv.org/abs/1905.12429
    
    Input:
        model: trained model
        test_dataset: test_dataset
        batch_size: batch size. Defaults to 512.
    """
    def __init__(self, model, test_dataset, batch_size = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.test_dataset = test_dataset

    def calc_class_distances(self):
        self.model.eval()
        data_loader_x = torch.utils.data.DataLoader(ClassDistDataset(self.test_dataset), batch_size=self.batch_size, drop_last =True)
        data_loader_xt = torch.utils.data.DataLoader(ClassDistDataset(self.test_dataset), batch_size=self.batch_size, drop_last =True)

        in_class = torch.empty(len(self.test_dataset)-len(self.test_dataset)%self.batch_size)

        out_class = torch.empty(len(self.test_dataset)-len(self.test_dataset)%self.batch_size)
        pb = tqdm(data_loader_x)
        for i, (idx, (x, target)) in enumerate(pb):
            x, target = Variable(x).to(self.device), Variable(target).to(self.device)
            min_dist_in = torch.FloatTensor([1e6]).repeat(x.shape[0]).to(self.device)
            min_dist_out = torch.FloatTensor([1e6]).repeat(x.shape[0]).to(self.device)
            with torch.no_grad():
                concepts_x = self.model.conceptizer(x)
                if isinstance(concepts_x, tuple):
                    concepts_x = concepts_x[0]

            for idxt, (xt, targets_t) in data_loader_xt:
                if not torch.equal(idx, idxt):
                    xt, targets_t = Variable(xt).to(self.device), Variable(targets_t).to(self.device)
                
                    with torch.no_grad():
                        concepts_xt = self.model.conceptizer(xt)
                        if isinstance(concepts_xt, tuple):
                            concepts_xt = concepts_xt[0]

                    dist = torch.cdist(concepts_x, concepts_xt)

                    inds_eq = target.reshape(-1, 1).repeat(1, xt.shape[0]) == targets_t.repeat(xt.shape[0], 1)

                    dist_in = dist.clone()
                    dist_out = dist.clone()
                    dist_out[inds_eq] = 1e6
                    dist_in[torch.logical_not(inds_eq)] = 1e6
                    min_dist_in = torch.min(torch.stack([min_dist_in, torch.min(dist_in, -1).values], 1), -1).values
                    min_dist_out = torch.min(torch.stack([min_dist_out, torch.min(dist_out, -1).values], 1), -1).values
            in_class[int(i*x.shape[0]):int(i*x.shape[0]+x.shape[0])] = min_dist_in
            out_class[int(i*x.shape[0]):int(i*x.shape[0]+x.shape[0])] = min_dist_out
            pb.set_postfix({"in class dist:": torch.mean(in_class[:int(i*x.shape[0]+x.shape[0])]), "out class dist:": torch.mean(out_class[:int(i*x.shape[0]+x.shape[0])])})
        self.in_class_distances = min_dist_in
        self.out_class_distances = min_dist_out

    def _pgd_out_class_dist(self, x, target_img, lmbda=-1e-4, k=55, eps=0.2, eps_step=2.5 * 0.2 / 55):
        with torch.no_grad():
            _, (concepts_target, _) = self.model(target_img)[0:2]
            x_adv = x.detach() + eps * (2 * torch.rand_like(x) - 1)
            pred_label = self.model(x)[0]
            pred_label = pred_label.argmax(1)
            for step in range(k):
                x_adv.detach_().requires_grad_()
                with torch.enable_grad():
                    pred, (concepts, _) = self.model(x_adv)[0:2]
                    self.model.zero_grad()
                    loss = ((concepts-concepts_target).norm() + lmbda * F.nll_loss(pred, pred_label)).backward()
                step = eps_step * x_adv.grad.sign()
                x_adv = x + (x_adv - step - x).clamp(min=-eps, max=eps)
        return concepts[pred.argmax(axis=1) == pred_label,:]

    def calc_adv_out_class_dist(self, early_stop=1000):
        data_loader_x = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        out_class = torch.empty(0).to(self.device)
        pb = tqdm(data_loader_x)
        for i, (x, target) in enumerate(pb):
            x, target = Variable(x).to(self.device), Variable(target).to(self.device)
            with torch.no_grad():
                concepts_x = self.model.conceptizer(x)[0]
            ## sample 100 images
            data_loader_xt = torch.utils.data.DataLoader(AdvOutDistEvalDataset(self.test_dataset, target), batch_size=100)
            min_dist_out = torch.FloatTensor([1e6]).to(self.device)
            xt, target_t = next(iter(data_loader_xt))
            xt, target_t = Variable(xt).to(self.device),Variable(target_t).to(self.device)
            concepts_xt = self._pgd_out_class_dist(xt, x)
            dist = (concepts_x-concepts_xt).norm(dim=1)
            if len(dist) != 0:
                min_dist_out = torch.min(dist).unsqueeze(0)
                out_class =  torch.cat((out_class, min_dist_out), 0)
            pb.set_postfix({"adv out dist:": torch.mean(out_class)})
            if i > early_stop:
                break
        self.adv_out_class_distances = out_class
            
if __name__ == "__main__":
    pass
