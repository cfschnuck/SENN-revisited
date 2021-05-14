import torch.optim as optim
import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from tqdm import tqdm
import copy
from abc import abstractmethod
import shutil
import os
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import logging, logging.handlers
from datetime import datetime
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# locals
from .losses import *
from .utils import *
from .eval_utils import *
class Trainer():
    """Trainer metaclass

    """
    def __init__(self, model, batch_size=132, lr=1e-3, dataset="MNIST", warm_start=False, path_pretrained=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.lr = lr
        self.warm_start = warm_start
        self.path_pretrained = path_pretrained
        self.dataset = dataset
        self.lipschitz = LipschitzEvaluator(self.model, eps=0.01)
        self.load_data()
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        self.time = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        self.model_path = dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{self.time}/"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logging.basicConfig(filename=dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{self.time}/train.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
        if self.path_pretrained is not None:
            self.load_checkpoint(self.path_pretrained)
        
    def load_data(self):
        transformers = {
            "MNIST" : torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
            "CIFAR10" : torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                #torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomCrop(32, padding=4), 
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
            "FashionMNIST" : torchvision.transforms.Compose([
                torchvision.transforms.Resize(32),
                torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
        }
        self.train_dataset = getattr(torchvision.datasets, self.dataset)("./Data/", train=True, download=True, transform=transformers[self.dataset])
        self.test_dataset = getattr(torchvision.datasets, self.dataset)("./Data/", train=False, download=True, transform=transformers[self.dataset])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
    
    @abstractmethod
    def propagate_forward(self):
        self.pred = None
    
    @abstractmethod
    def calculate_loss(self):
        self.total_loss = None
    
    @property
    def model_params(self):
        return {"lr" : self.lr, "batch_size": self.batch_size}

    def _log_epoch(self, epoch):
        logging.info(f"Epoch: {epoch} | Total Loss: {self.total_loss_meter.mean} | Acc: {self.acc_meter.mean}")

    def accuracy(self, pred, y):
        return torch.mean((pred.argmax(axis=1) == y).float()).item()
    
    def epoch_init(self):
        pass
    
    def train(self, n_epochs = 10):
        best_score = 0
        for epoch in range(1, n_epochs+1):
            self.epoch_init() 
            self.acc_meter = AverageValueMeter()
            self.total_loss_meter = AverageValueMeter() 
            self.model.train()
            for batch_id, (x, targets) in enumerate(tqdm(self.train_loader, desc="Train: ")):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                self.x.requires_grad_(True)
                self.propagate_forward()
                self.calculate_loss()
                self.optimizer.zero_grad()
                self.total_loss.backward()
                self.optimizer.step()
                self.acc_meter.add(self.accuracy(self.pred, self.targets))
                self.total_loss_meter.add(self.total_loss.item())
                # if batch_id ==10:
                #     break
            self._log_epoch(epoch)
            self.validate()
            is_best = self.acc_meter_val.mean > best_score
            best_score = max(self.acc_meter_val.mean, best_score)
            self.epoch_end(epoch)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'params' : self.model_params
                    }, is_best)
            if epoch % 10 == 0:
                lip = self.lipschitz.dataset_lipschitz_estimate(self.test_dataset, 100)
                logging.info(f"Lipschitz: {lip}")

    def epoch_end(self, epoch):
        pass

    def validate(self):
        self.model.eval()
        self.acc_meter_val = AverageValueMeter()
        self.adv_acc_meter_val = AverageValueMeter()
        with torch.no_grad():
            for x, targets in tqdm(self.test_loader, desc="Test: "):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                self.propagate_forward()
                x_adv = pgd(self.model, self.x, self.targets, k=7, eps=0.1, eps_step=2.5 * 0.1 / 7)
                pred_adv = self.model(x_adv)[0]
                self.adv_acc_meter_val.add(self.accuracy(pred_adv, self.targets))
                self.acc_meter_val.add(self.accuracy(self.pred, self.targets))
            logging.info(f"Acc: {self.acc_meter_val.mean} | Adv Acc: {self.adv_acc_meter_val.mean}")
    
    def save_checkpoint(self, state, is_best):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        filename = self.model_path + 'checkpoint.pth.tar.gz'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.model_path + 'model_best.pth.tar.gz')
    
    def load_checkpoint(self, path):
        try:
            self.state_checkpoint = torch.load(path + 'checkpoint.pth.tar.gz', map_location=self.device)
            self.state_best_model = torch.load(path + 'model_best.pth.tar.gz', map_location=self.device)
        except FileNotFoundError:
            logging.warning("No pretrained model found. Training continues without pretrained weights.")
        else:
            self.model.load_state_dict(self.state_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.state_checkpoint['optimizer_state_dict'])
            self.best_model = copy.deepcopy(self.model)
            self.best_model.load_state_dict(self.state_best_model['model_state_dict'])
            self.best_model.model_params = self.state_best_model['params']
    
    def pca(self, concepts, targets, epoch):
        concepts = concepts.detach().cpu().numpy()
        y = targets.cpu().numpy()
        pca = PCA(n_components=2)
        X = pca.fit(concepts).transform(concepts)
        plt.figure()
        for i in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
            plt.scatter(X[y == i, 0], X[y == i, 1], s=2, alpha=.6,
                label=str(i))
        plt.title("PCA for " + self.model.__class__.__name__ + " on " + self.dataset + " (" + str(self.robustness_reg) + ")")
        plt.tight_layout()
        plt.savefig(self.model_path + f'pca_concepts_epoch_{epoch}.png')
    
    def tsne(self, concepts, targets, epoch):
        concepts = concepts.detach().cpu().numpy()
        y = targets.cpu().numpy()
        Z = TSNE(n_components=2).fit_transform(concepts)
        plt.figure()
        for i in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
            plt.scatter(Z[y == i, 0], Z[y == i, 1], s=2, alpha=.6,
            label=str(i))
        plt.title("TSNE for " + self.model.__class__.__name__ + " on " + self.dataset + " (" + str(self.robustness_reg) + ")")
        plt.tight_layout()
        plt.savefig(self.model_path + f'tsne_concepts_epoch_{epoch}.png')

    def bar_plot(self):
        """Create bar plot for distances

        Args:
            in_dist: in-class distance
            out_dist: out-class distance
            adv_out_dist: adv out-class distance
            plt_path: file path for saving 
        """
        d = DistanceEvaluator(self.model, self.test_dataset)
        d.calc_class_distances()
        d.calc_adv_out_class_dist(2)

        data = [d.adv_out_class_distances.detach().cpu(), d.in_class_distances.detach().cpu(), d.out_class_distances.detach().cpu()]
        fig, ax = plt.subplots()
        bp = ax.boxplot(data, patch_artist=True,  showmeans=False, showfliers=False, labels = ["Adv. Out-class", "In-class", "Out-class"])
    
    
        for element in ['boxes', 'whiskers', 'medians', 'caps']:
            plt.setp(bp[element], color="blue")

        for patch in bp['boxes']:
            patch.set(facecolor="cyan")  
    
        ax.set_ylabel('Minimum h(x) distance')
        plt.savefig(self.model_path + f'boxplot_distances.png')

class VanillaSennTrainer(Trainer):
    """Trainer for SENN

    Args:
        model: model architecture
        robustness_reg: robustness penalty lambda
        concept_reg: reconstruction penalty in conceptizer
        sparsity_reg: sparsity penalty for concepts
        batch_size: batch size
        lr: learning rate
        dataset: dataset. Defaults to MNIST.
        warm_start: load pretrained model if availale. Defaults to True.
        path_pretrained: path to a pretrained model

    """
    def __init__(self, model, robustness_reg = 2e-4, concept_reg = 1, sparsity_reg = 2e-5, batch_size=132, lr=2e-4, dataset="MNIST", warm_start=True, path_pretrained=None):
        super(VanillaSennTrainer, self).__init__(model, batch_size=batch_size, lr=lr, dataset=dataset, warm_start=warm_start, path_pretrained=path_pretrained)
        self.robustness_reg = robustness_reg
        self.concept_reg = concept_reg
        self.sparsity_reg = sparsity_reg
        self.concept_loss = mse_concepts_sparsity_loss
        self.classification_loss = F.nll_loss
        self.robustness_loss = simple_robustness_loss
        self.lipschitz = LipschitzEvaluator(self.model, eps=0.01)
        logging.info(self.model_params)
    
    @property
    def model_params(self):
        return {"lr" : self.lr, "batch_size": self.batch_size, "robustness_reg" : self.robustness_reg, "concept_reg": self.concept_reg, "sparsity_reg": self.sparsity_reg}
    
    def propagate_forward(self):
        self.pred, (self.concepts, self.relevances), self.x_reconstructed = self.model(self.x)
    
    def calculate_loss(self):
        classification_loss = self.classification_loss(self.pred, self.targets)
        robustness_loss = self.robustness_loss(self.x, self.pred, self.concepts, self.relevances)
        concept_loss = self.concept_loss(self.x, self.x_reconstructed, self.concepts, self.sparsity_reg)
        self.total_loss = classification_loss + self.robustness_reg * robustness_loss + self.concept_reg * concept_loss
        self.robustness_loss_meter.add(robustness_loss.item())
        self.classification_loss_meter.add(classification_loss.item())
        self.concept_loss_meter.add(concept_loss.item())
        
    def epoch_init(self):
        self.concept_loss_meter = AverageValueMeter()
        self.robustness_loss_meter = AverageValueMeter()
        self.classification_loss_meter = AverageValueMeter()
    
    def epoch_end(self, epoch):
        if epoch % 50 == 0:
            lip = self.lipschitz.dataset_lipschitz_estimate(self.test_dataset, 100)
            logging.info(f"Lipschitz: {lip}")
        if epoch % 25 == 0:
            self.model.eval()
            d_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4000, shuffle=True)
            x, targets = next(iter(d_loader))
            with torch.no_grad():
                _, (concepts, _), _ = self.model(x.to(self.device))
            self.pca(concepts, targets, epoch)
            self.tsne(concepts, targets, epoch)
        if epoch % 3 == 0:
            d = DistanceEvaluator(self.model, self.test_dataset)
            d.calc_class_distances()
            logging.info(f"in class: {torch.mean(d.in_class_distances)}")
            logging.info(f"out class: {torch.mean(d.out_class_distances)}")
            d.calc_adv_out_class_dist()
            logging.info(f"adv out class: {torch.mean(d.adv_out_class_distances)}")
    
    def _log_epoch(self, epoch):
        logging.info(f"Epoch: {epoch} | Total Loss: {self.total_loss_meter.mean} | Concept Loss: {self.concept_loss_meter.mean} | Robustness Loss: {self.robustness_loss_meter.mean} | Clf Loss: {self.classification_loss_meter.mean} | Acc: {self.acc_meter.mean}")

class VAETrainerSeperated(Trainer):
    """Trainer for VaeSENN

    Args:
        model: model architecture
        robustness_reg: robustness penalty lambda
        beta_reg_styles: beta perameter in style vae
        beta_reg_concepts: beta perameter in conceptizer
        batch_size: batch size
        lr: learning rate
        dataset: dataset. Defaults to MNIST.
        warm_start: load pretrained model if availale. Defaults to True.
        path_pretrained: path to a pretrained model
        pretrain: if pretrain conceptizer will be pretrained
    """
    def __init__(self, model, robustness_reg = 1e-4, beta_reg_styles = 1e-2,beta_reg_concepts=1e-2,  batch_size=132, lr=2e-4, dataset="MNIST", warm_start=True, pretrain=False, path_pretrained=None):
        super(VAETrainerSeperated, self).__init__(model, batch_size=batch_size, lr=lr, dataset=dataset, warm_start=warm_start, path_pretrained=path_pretrained)
        self.robustness_reg = robustness_reg
        self.beta_reg_styles = beta_reg_styles
        self.beta_reg_concepts = beta_reg_concepts
        self.classification_loss = F.nll_loss
        self.robustness_loss = simple_robustness_loss
        self.recon_loss_styles = F.mse_loss
        self.recon_loss_concepts = F.mse_loss
        self.pretrain = pretrain
        self.kl_div = kl_div
        self.optimizer_vae_styles = optim.Adam(list(self.model.conceptizer.encoder_styles.parameters())+list(self.model.conceptizer.decoder_styles.parameters()), lr= self.lr)
        self.optimizer_senn = optim.Adam(list(self.model.conceptizer.encoder_concepts.parameters())+list(self.model.conceptizer.decoder_concepts.parameters())+list(self.model.parametrizer.parameters()), lr= self.lr)
        if self.pretrain: 
            self.opimizer_conceptizer = optim.Adam(list(self.model.conceptizer.encoder_concepts.parameters())+list(self.model.conceptizer.decoder_concepts.parameters()), lr= self.lr)
        self.lipschitz = LipschitzEvaluator(self.model)
        logging.info(self.model_params)

    @property
    def model_params(self):
        return {"lr" : self.lr, "batch_size": self.batch_size, "robustness_reg" : self.robustness_reg, "beta_reg_concepts": self.beta_reg_concepts, "beta_reg_styles": self.beta_reg_styles}

    def train(self, n_epochs_vae_styles = 20, n_epochs_senn=50):
        best_score = 0
        ## train style senn
        for epoch in range(1, n_epochs_vae_styles+1):
            self.model.train()
            self.styles_recon_loss_meter = AverageValueMeter()
            self.styles_kl_div_meter = AverageValueMeter()
            for batch_id, (x, targets) in enumerate(tqdm(self.train_loader, desc="Train: ")):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                ## train style vae
                self.model.zero_grad()
                z, mean, log_var, decoded_styles = self.model.conceptizer.forward_styles(self.x, self.targets)
                recon_loss = self.recon_loss_styles(self.x, decoded_styles)
                kl_div = self.kl_div(mean, log_var)
                vae_loss = recon_loss + self.beta_reg_styles * kl_div
                vae_loss.backward()
                self.optimizer_vae_styles.step()
                self.styles_recon_loss_meter.add(recon_loss.item())
                self.styles_kl_div_meter.add(kl_div.item())
            self._log_epoch_styles(epoch)
        ## freeze weights
        for param in self.model.conceptizer.encoder_styles.parameters():
            param.requires_grad = False
        for param in self.model.conceptizer.decoder_styles.parameters():
            param.requires_grad = False
        ## trainer conceptizer
        if self.pretrain: 
            for epoch in range(1, n_epochs_vae_styles+1):
                self.model.train()
                self.styles_recon_loss_meter = AverageValueMeter()
                self.styles_kl_div_meter = AverageValueMeter()
                for batch_id, (x, targets) in enumerate(tqdm(self.train_loader, desc="Train: ")):
                    self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                    ## train style vae
                    self.model.zero_grad()
                    pred, (concepts, relevances), x_recon, log_var, mean = self.model(self.x)
                    recon_loss = self.recon_loss_styles(self.x, x_recon)
                    kl_div = self.kl_div(mean, log_var)
                    vae_loss = recon_loss + kl_div
                    vae_loss.backward()
                    self.opimizer_conceptizer.step()
                    self.styles_recon_loss_meter.add(recon_loss.item())
                    self.styles_kl_div_meter.add(kl_div.item())
                self._log_epoch_styles(epoch)
        ## train full model
        for epoch in range(1, n_epochs_senn+1): 
            self.model.train()
            self.acc_meter = AverageValueMeter()
            self.concept_recon_loss_meter = AverageValueMeter()
            self.concept_kl_div_meter = AverageValueMeter()
            self.robustness_loss_meter = AverageValueMeter()
            self.classification_loss_meter = AverageValueMeter()
            self.total_loss_meter = AverageValueMeter()
            for batch_id, (x, targets) in enumerate(tqdm(self.train_loader, desc="Train: ")):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                self.x.requires_grad_(True)
                self.model.zero_grad()
                pred, (concepts, relevances), x_recon, log_var, mean = self.model(self.x)
                classification_loss = self.classification_loss(pred, self.targets)
                robustness_loss = self.robustness_loss(self.x, pred, concepts, relevances)
                recon_loss_concepts = self.recon_loss_concepts(self.x.detach(), x_recon)
                kl_div = self.kl_div(mean, log_var)
                total_loss = classification_loss + self.robustness_reg * robustness_loss + recon_loss_concepts + self.beta_reg_concepts * kl_div
                total_loss.backward()
                self.optimizer_senn.step()
                self.acc_meter.add(self.accuracy(pred, self.targets))
                self.total_loss_meter.add(total_loss.item())
                self.concept_recon_loss_meter.add(recon_loss_concepts.item())
                self.robustness_loss_meter.add(robustness_loss.item())
                self.concept_kl_div_meter.add(kl_div.item())
                self.classification_loss_meter.add(classification_loss.item())
            self._log_epoch_full(epoch)
            
            if epoch % 25 == 0:
                d_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4000, shuffle=True)
                x, targets = next(iter(d_loader))
                with torch.no_grad():
                    _, (concepts, _), _, _ = self.model(x.to(self.device))
                self.tsne(concepts, targets, epoch)
                self.pca(concepts, targets, epoch)

            self.validate()
            is_best = self.acc_meter_val.mean > best_score
            best_score = max(self.acc_meter_val.mean, best_score)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'train acc': self.acc_meter.mean,
                    'val acc': self.acc_meter_val.mean,
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'total_loss' : total_loss,
                    'params' : self.model_params
                    }, is_best)
            if epoch % 50 == 0:
                lip = self.lipschitz.dataset_lipschitz_estimate(self.test_dataset, 100)
                logging.info(f"Lipschitz: {lip}")

    def validate(self):
        self.model.eval()
        self.acc_meter_val_adv = AverageValueMeter()
        self.acc_meter_val = AverageValueMeter()
        with torch.no_grad():
            for x, targets in tqdm(self.test_loader, desc="Test: "):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                pred, (concepts, relevances), x_recon, log_var = self.model(self.x)
                self.acc_meter_val.add(self.accuracy(pred, self.targets))
                x_adv = pgd(self.model, self.x, self.targets, k=7, eps=0.1, eps_step=2.5 * 0.1 / 7)
                pred_adv, _, _, _ = self.model(x_adv)
                self.acc_meter_val_adv.add(self.accuracy(pred_adv, self.targets))
            logging.info(f"Acc: {self.acc_meter_val.mean} | Adv acc: {self.acc_meter_val_adv.mean}")
    
    def _log_epoch_styles(self, epoch):
        logging.info(f"Style VAE: Epoch: {epoch} | Recon Loss Styles: {self.styles_recon_loss_meter.mean} | Kl div: {self.styles_kl_div_meter.mean}")

    def _log_epoch_full(self, epoch):
        logging.info(f"Full model: Epoch: {epoch} | Total Loss: {self.total_loss_meter.mean} | Recon Loss Concepts: {self.concept_recon_loss_meter.mean} | Robustness Loss: {self.robustness_loss_meter.mean} | Clf Loss: {self.classification_loss_meter.mean} | Kl div: {self.concept_kl_div_meter.mean} | Acc: {self.acc_meter.mean}")

class GaussTripletSiamSennTrainer(Trainer):
    """Trainer for VSiamSenn

    Args:
        model: model architecture
        robustness_reg: robustness penalty lambda
        simsiam_reg: penalty for siamese loss
        rob_reg_concepts: robustness penalty eta
        beta_reg: beta parameter of conceptizer
        batch_size: batch size
        lr: learning rate
        dataset: dataset. Defaults to MNIST.
        warm_start: load pretrained model if availale. Defaults to True.
        path_pretrained: path to a pretrained model

    """
    def __init__(self, model, robustness_reg = 2e-4, simsiam_reg = 1, rob_reg_concepts= 1e-2, beta_reg=1e-3, batch_size=132, lr=2e-4, dataset="MNIST", warm_start=True, path_pretrained=None):
        super(GaussTripletSiamSennTrainer, self).__init__(model, batch_size=batch_size, lr=lr, dataset=dataset, warm_start=warm_start, path_pretrained=path_pretrained)
        self.robustness_reg = robustness_reg
        self.rob_reg_concepts = rob_reg_concepts
        self.simsiam_reg = simsiam_reg
        self.classification_loss = F.nll_loss
        self.beta_reg = beta_reg
        self.robustness_loss = simple_robustness_loss
        self.lipschitz = LipschitzEvaluator(self.model)
        logging.info(self.model_params)
        self.optimizer_parametrizer = optim.Adam(self.model.parametrizer.parameters(), lr = self.lr)
        self.optimizer_conceptizer = optim.Adam(self.model.conceptizer.parameters(), lr = self.lr*10)

    def train(self, n_epochs):
        best_score = 0
        ## train siam senn 
        self.tripletsimsiam_dataset = TripletSimSiamDataset(self.train_dataset)
        self.tripletsimsiam_train_loader = torch.utils.data.DataLoader(self.tripletsimsiam_dataset, batch_size=self.batch_size, shuffle=True)
        
        best_score = 0
        for epoch in range(1, n_epochs+1):
            self.model.train()
            self.pos_loss_meter = AverageValueMeter()
            self.neg_loss_meter = AverageValueMeter()
            self.acc_meter = AverageValueMeter()
            self.robustness_loss_meter = AverageValueMeter()
            self.classification_loss_meter = AverageValueMeter()
            self.kl_loss_meter = AverageValueMeter()
            self.total_loss_meter = AverageValueMeter()

            for (x1, targets1), (x2, targets2), (x3, targets3) in tqdm(self.tripletsimsiam_train_loader, desc="Train: ", total=len(self.train_loader)):
                self.model.zero_grad()
                self.x1, self.targets1, self.x2, self.targets2, self.x3, self.targets3 = Variable(x1).to(self.device), Variable(targets1).to(self.device), Variable(x2).to(self.device), Variable(targets2).to(self.device), Variable(x3).to(self.device), Variable(targets3).to(self.device)
                self.x1.requires_grad_(True)
                pred, (concepts, relevances), (L1, L2, KL) = self.model(self.x1, self.x2, self.x3)
                simsiam_loss = L1 + L2 + self.beta_reg * KL
                classification_loss = self.classification_loss(pred, self.targets1)
                robustness_loss = self.robustness_loss(self.x1, pred, concepts, relevances)
                total_loss = classification_loss + self.robustness_reg * robustness_loss + self.simsiam_reg * simsiam_loss + self.rob_reg_concepts * compute_jacobian(self.x1, concepts).norm()
                total_loss.backward()
                self.optimizer_parametrizer.step()
                self.optimizer_conceptizer.step()
                self.pos_loss_meter.add(L1.item())
                self.neg_loss_meter.add(L2.item())
                self.kl_loss_meter.add(KL.item())
                self.classification_loss_meter.add(classification_loss.item())
                self.robustness_loss_meter.add(robustness_loss.item())
                self.total_loss_meter.add(total_loss.item())
                self.acc_meter.add(self.accuracy(pred, self.targets1))
            self._log_epoch_full_model(epoch)
            
            if epoch % 25 == 0:
                self.model.eval()
                d_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1000, shuffle=True)
                x, targets = next(iter(d_loader))
                with torch.no_grad():
                    _, (concepts, _) = self.model(x.to(self.device))
                self.pca(concepts, targets, epoch)
                self.tsne(concepts, targets, epoch)
            
            self.validate()
            is_best = self.acc_meter_val.mean > best_score
            best_score = max(self.acc_meter_val.mean, best_score)
            self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'params' : self.model_params
                    }, is_best)
            if epoch % 50 == 0:
                lip = self.lipschitz.dataset_lipschitz_estimate(self.test_dataset, 100)
                logging.info(f"Lipschitz: {lip}")
            if epoch % 100 == 0:
                self.bar_plot()
    @property
    def model_params(self):
        return {"robustness_reg" : self.robustness_reg, "batch_size": self.batch_size, "lr": self.lr, "simsiam_reg" : self.simsiam_reg, "beta_reg": self.beta_reg, "rob_reg_concepts": self.rob_reg_concepts}
    
    def _log_epoch_full_model(self, epoch):
        logging.info(f"Senn: Epoch: {epoch} | Clf Loss: {self.classification_loss_meter.mean} | Robustness Loss: {self.robustness_loss_meter.mean} | Pos loss: {self.pos_loss_meter.mean} | Neg Loss: {self.neg_loss_meter.mean} | Total loss: {self.total_loss_meter.mean} | KL: {self.kl_loss_meter.mean} | Acc: {self.acc_meter.mean}")
    
    def validate(self):
        self.model.eval()
        self.acc_meter_val = AverageValueMeter()
        self.adv_acc_meter_val = AverageValueMeter()
        for x, targets in tqdm(self.test_loader, desc="Test: "):
            self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
            pred, (concepts, relevances) = self.model(self.x)
            self.acc_meter_val.add(self.accuracy(pred, self.targets))
            x_adv = pgd(self.model, self.x, self.targets, k=7, eps=0.1, eps_step=2.5 * 0.1 / 7)
            pred_adv, _ = self.model(x_adv)
            self.adv_acc_meter_val.add(self.accuracy(pred_adv, self.targets))
        logging.info(f"Acc: {self.acc_meter_val.mean} | Adv Acc: {self.adv_acc_meter_val.mean}")

class InvarSennTrainer(Trainer):
    """Trainer for InvarSenn

    Args:
        model: model architecture
        robustness_reg: robustness penalty lambda
        concepts_sparsity_reg: sparsity penalty on concepts
        recon_reg_concepts: reconstruction penalty on concepts
        disentangle_reg: disentangle penalty
        update_ratio: update ratio
        disentangle_patience: patience in disentanglement
        batch_size: batch size
        lr: learning rate
        dataset: dataset. Defaults to MNIST.
        warm_start: load pretrained model if availale. Defaults to True.
        path_pretrained: path to a pretrained model

    """
    def __init__(self, model, robustness_reg, concepts_sparsity_reg = 0, recon_reg_concepts=1e-5, disentangle_reg = 1e-2, update_ratio = 200, disentangle_patience=5, batch_size=132, lr_m1=2e-4, lr_m2=2e-3, dataset="MNIST", warm_start=True, path_pretrained=None):
        super(InvarSennTrainer, self).__init__(model, batch_size=batch_size, lr=0, dataset=dataset)
        self.lr_m1 = lr_m1
        self.lr_m2 = lr_m2
        self.optimizer1 = optim.Adam(self.model.m1.parameters(), lr= self.lr_m1)
        self.optimizer2 = optim.Adam(self.model.m2.parameters(), lr= self.lr_m2)
        self.robustness_reg = robustness_reg
        self.concepts_sparsity_reg = concepts_sparsity_reg
        self.recon_reg_concepts = recon_reg_concepts
        self.disentangle_reg = disentangle_reg
        self.update_ratio = update_ratio
        self.disentangle_patience = disentangle_patience
        self.classification_loss = F.nll_loss
        self.robustness_loss = simple_robustness_loss
        self.recon_loss_concepts = mse_concepts_sparsity_loss
        self.disentangle_loss = disentangle_loss
        logging.info(self.model_params)
        self.lipschitz = LipschitzEvaluator(self.model, eps=0.01)
        self.path_pretrained = path_pretrained
        self.warm_start = warm_start
        if self.warm_start:
            if self.path_pretrained is None:
                logging.warning("No path to pretrained model specified. Last model used.")
                last_training = sorted(os.listdir(dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/"), reverse=True)[0]
                path = dirname(dirname(realpath(__file__))) + f"/Pretrained/{self.dataset}/{self.model.__class__.__name__}/{last_training}/"
                self.load_checkpoint(path)
            else:
                self.load_checkpoint(self.path_pretrained)


    @property
    def model_params(self):
        return {"lr" : self.lr, "batch_size": self.batch_size, "robustness_reg" : self.robustness_reg, "recon_reg_concepts" : self.recon_reg_concepts, "concepts_sparsity_reg": self.concepts_sparsity_reg, "disentangle_reg" : self.disentangle_reg, "update_ratio" : self.update_ratio}
    

    def train(self, n_epochs = 20):
        best_score = 0
        for epoch in range(1, n_epochs+1):
            self.acc_meter = AverageValueMeter()
            self.concept_recon_loss_meter = AverageValueMeter()
            self.robustness_loss_meter = AverageValueMeter()
            self.classification_loss_meter = AverageValueMeter()
            self.disentangle_loss1_meter = AverageValueMeter()
            self.total_loss_meter = AverageValueMeter()
            # train M1
            self.model.m1.train()
            self.model.m2.eval()
            self._freeze_weights_m1(unfreeze=True)
            self._freeze_weights_m2(unfreeze=False)
            e_epoch = []
            for batch_id, (x, targets) in enumerate(tqdm(self.train_loader, desc="Train: ")):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                self.x.requires_grad_(True)
                self.model.zero_grad()
                pred, (e1, relevances), e2, x_reconstructed = self.model.m1(self.x)
                classification_loss = self.classification_loss(pred, self.targets)
                robustness_loss = self.robustness_loss(self.x, pred, e1, relevances)
                recon_loss_concepts = self.recon_loss_concepts(self.x.detach(), x_reconstructed, e1, self.concepts_sparsity_reg)
                e1_reconstructed, e2_reconstructed = self.model.m2(e1, e2)
                e1_random, e2_random = sample_from_latent(e1.size(), e2.size())
                disentangle_loss1 = self.disentangle_loss(e1_random.to(self.device), e1_reconstructed, e2_random.to(self.device), e2_reconstructed)
                total_loss = classification_loss + self.robustness_reg * robustness_loss + self.recon_reg_concepts * recon_loss_concepts + self.disentangle_reg * disentangle_loss1
                total_loss.backward()
                self.optimizer1.step()
                self.acc_meter.add(self.accuracy(pred, self.targets))
                self.total_loss_meter.add(total_loss.item())
                self.concept_recon_loss_meter.add(recon_loss_concepts.item())
                self.robustness_loss_meter.add(robustness_loss.item())
                self.classification_loss_meter.add(classification_loss.item())
                self.disentangle_loss1_meter.add(disentangle_loss1.item())
                e_epoch.append((e1.detach().clone(), e2.detach().clone()))
            # train M2
            self._freeze_weights_m1(unfreeze=False)
            self._freeze_weights_m2(unfreeze=True)
            e_epoch_train, e_epoch_test = self.e_epoch_split(e_epoch)
            i = 0
            best_loss = 1e6
            no_improvement_count = 0
            while i <= self.update_ratio and no_improvement_count <= self.disentangle_patience:
                i += 1
                self.disentangle_loss2_meter = AverageValueMeter()
                self.disentangle_loss2_val_meter = AverageValueMeter()
                self.model.m2.train()
                self.model.m1.eval()
                for batch_id, (e1, e2) in enumerate(e_epoch_train):
                    self.e1, self.e2 = Variable(e1).to(self.device), Variable(e2).to(self.device)
                    self.e1.requires_grad_(True)
                    self.e2.requires_grad_(True)
                    self.model.zero_grad()
                    e1_reconstructed, e2_reconstructed = self.model.m2(self.e1, self.e2)
                    disentangle_loss2 = self.disentangle_loss(self.e1, e1_reconstructed, self.e2, e2_reconstructed)
                    disentangle_loss2.backward()
                    self.optimizer2.step()
                    self.disentangle_loss2_meter.add(disentangle_loss2.item())
                self.model.m2.eval()
                self.disentangle_acc_meter_val = AverageValueMeter()
                with torch.no_grad():
                    for batch_id, (e1, e2) in enumerate(e_epoch_test):
                        self.e1, self.e2 = Variable(e1).to(self.device), Variable(e2).to(self.device)
                        e1_reconstructed, e2_reconstructed = self.model.m2(self.e1, self.e2)
                        disentangle_loss2_val =  self.disentangle_loss(self.e1, e1_reconstructed, self.e2, e2_reconstructed)
                        self.disentangle_loss2_val_meter.add(disentangle_loss2_val.item())
                best_loss = min(best_loss, self.disentangle_loss2_val_meter.mean+1e-3)
                if i > 2 and best_loss < self.disentangle_loss2_val_meter.mean+1e-3:
                    no_improvement_count += 1
            logging.info(f"Updates: {i-1} | Disentangle Loss 2: {self.disentangle_loss2_meter.mean} | Test Disentangle Loss 2: {self.disentangle_loss2_val_meter.mean}")
                    
                        

            self._log_epoch_full(epoch)
            self.validate()
            is_best = self.acc_meter_val.mean > best_score
            best_score = max(self.acc_meter_val.mean, best_score)
            self.save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train acc': self.acc_meter.mean,
                    'val acc': self.acc_meter_val.mean,
                    'optimizer1_state_dict' : self.optimizer1.state_dict(),
                    'optimizer2_state_dict' : self.optimizer2.state_dict(), 
                    'total_loss' : total_loss,
                    'params' : self.model_params
                    }, is_best)
            if epoch % 10 == 0:
                self._freeze_weights_m1(unfreeze=False)
                self._freeze_weights_m2(unfreeze=False)
                lip = self.lipschitz.dataset_lipschitz_estimate(self.test_dataset, 100)
                logging.info(f"Lipschitz: {lip}")


    def validate(self):
        self.model.eval()
        self.acc_meter_val = AverageValueMeter()
        self.disentangle_acc_meter_val = AverageValueMeter()
        self.adv_acc_meter_val = AverageValueMeter()
        with torch.no_grad():
            for x, targets in tqdm(self.test_loader, desc="Test: "):
                self.x, self.targets = Variable(x).to(self.device), Variable(targets).to(self.device)
                pred, (e1, relevances), e2, x_reconstructed = self.model.m1(self.x)
                e1_reconstructed, e2_reconstructed = self.model.m2(e1, e2)
                self.acc_meter_val.add(self.accuracy(pred, self.targets))
                # calulate validation disentanglement accuracy
                disentangle_loss2_val =  self.disentangle_loss(e1, e1_reconstructed, e2, e2_reconstructed)
                self.disentangle_loss2_val_meter.add(disentangle_loss2_val.item())
                # calculate adversarial accuracy
                x_adv = pgd(self.model, self.x, self.targets, k=7, eps=0.1, eps_step=2.5 * 0.1 / 7)
                pred_adv = self.model(x_adv)[0]
                self.adv_acc_meter_val.add(self.accuracy(pred_adv, self.targets))
            logging.info(f"Test Acc: {self.acc_meter_val.mean} | Test Disentangle loss: {self.disentangle_loss2_val_meter.mean}  Adv Acc: {self.adv_acc_meter_val.mean}")
        
    def e_epoch_split(self, e_epoch):
        e_size = len(e_epoch)
        e_random = random.sample(e_epoch, e_size)
        return e_random[:int(0.8 * e_size)], e_random[int(0.8 * e_size):]

    def _log_epoch_full(self, epoch):
        logging.info(f"Full model: Epoch: {epoch} | Total Loss: {self.total_loss_meter.mean} | Recon Loss Concepts: {self.concept_recon_loss_meter.mean} | Robustness Loss: {self.robustness_loss_meter.mean} | Clf Loss: {self.classification_loss_meter.mean} | Disentangle Loss 1: {self.disentangle_loss1_meter.mean} |Disentangle Loss 2: {self.disentangle_loss2_meter.mean} | Acc: {self.acc_meter.mean}")

    def load_checkpoint(self, path):
        try:
            self.state_checkpoint = torch.load(path + 'checkpoint.pth.tar.gz', map_location=self.device)
            self.state_best_model = torch.load(path + 'model_best.pth.tar.gz', map_location=self.device)
        except FileNotFoundError:
            logging.warning("No pretrained model found. Training continues without pretrained weights.")
        else:
            self.model.load_state_dict(self.state_checkpoint['model_state_dict'])
            self.optimizer1.load_state_dict(self.state_checkpoint['optimizer1_state_dict'])
            self.optimizer2.load_state_dict(self.state_checkpoint['optimizer2_state_dict'])
            self.best_model = copy.deepcopy(self.model)
            self.best_model.load_state_dict(self.state_best_model['model_state_dict'])
            self.best_model.model_params = self.state_best_model['params']

    def _freeze_weights_m1(self, unfreeze):
        for param in self.model.m1.parameters():
            param.requires_grad = unfreeze

    def _freeze_weights_m2(self, unfreeze):
        for param in self.model.m2.parameters():
            param.requires_grad = unfreeze


if __name__ == "__main__":
    pass