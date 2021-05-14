import argparse

class Argparser():
    """Argparser class
        Args: 
            model name
        Returns:
            argparser for model corresponding to model name

    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--batch_size', type=int, default=132, help='batch size')
        self.parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
        self.parser.add_argument('--n_concepts', type=int, default=5, help='number of concepts')
        self.parser.add_argument('--robustness_reg', type=float, default=1e-4, help='robustness penalty for parametrizer')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
        self.parser.add_argument('--path_pretrained', type=str, default=None, help='path to pretrained model')
        if self.model_name == "senn":
            self._senn_arguments()
        elif self.model_name == "vaesenn":
            self._vaesenn_arguments()
        elif self.model_name == "simsiamsenn":
            self._siamsenn_arguments()
        elif self.model_name == "vsiamsenn":
            self._gausssiamsenn_arguments()
        elif self.model_name == "invarsenn":
            self._invarsenn_arguments()
        else:
            raise NotImplementedError
    

    def _senn_arguments(self):
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--sparsity_reg', type=float, default=2e-5, help='sparsity penalty for concepts')
        self.parser.add_argument('--recon_reg', type=float, default=1, help='weight of reconstruction loss of concept autoencoder')
    
    def _vaesenn_arguments(self):
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--beta_reg_styles', type=float, default=1e-2, help='beta parameter of styles vae')
        self.parser.add_argument('--beta_reg_concepts', type=float, default=1e-2, help='beta parameter of concepts vae')
        self.parser.add_argument('--n_styles', type=int, default=5, help='number of styles')
        self.parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of training epochs for styles')
    
    def _gausssiamsenn_arguments(self):
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--simsiam_reg', type=float, default=1., help='weight of loss of siamese network')
        self.parser.add_argument('--beta_reg', type=float, default=1e-3, help='beta parameter for kl divergence in siamese network')
        self.parser.add_argument('--robustness_reg_concepts', type=float, default=1e-4, help='robustness penalty for conceptizer')
    
    def _siamsenn_arguments(self):
        self.parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        self.parser.add_argument('--simsiam_reg', type=float, default=1., help='weight of loss of siamese network')
    
    def _invarsenn_arguments(self):
        self.parser.add_argument('--noise_dim', type=int, default=30, help='dimension of noise latent space')
        self.parser.add_argument('--recon_reg', type=float, default=1e-5, help='weight of reconstruction loss of concept autoencoder')
        self.parser.add_argument('--lr_m1', type=float, default=2e-4, help='learning rate m1')
        self.parser.add_argument('--lr_m2', type=float, default=2e-3, help='learning rate m2')
        self.parser.add_argument('--disentangle_reg', type=float, default=1e-2, help='disentanglement penalty')
        self.parser.add_argument('--disentangle_patience', type=int, default=5, help='disentanglement patience')
        self.parser.add_argument('--update_ratio', type=int, default=200, help='update ratio')

    def parse_args(self):
        return self.parser.parse_args()
