import argparse
import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

# Local imports
from SENN.models import InvarSennM
from SENN.invarsenn import InvarSennM1, InvarSennM2
from SENN.conceptizers import InvarConceptizer
from SENN.parametrizers import SENNParametrizer
from SENN.aggregators import BaseAggregator
from SENN.trainers import InvarSennTrainer
from SENN.disentanglers import Disentangler
from SENN.argparser import Argparser


parser = Argparser("invarsenn")
args = parser.parse_args()

def main():
    conceptizer = InvarConceptizer(args.n_concepts, args.noise_dim, args.dataset)
    aggregator = BaseAggregator()
    parametrizer = SENNParametrizer(args.n_concepts, args.dataset)
    disentangler1 = Disentangler(args.noise_dim, args.n_concepts)
    disentangler2 = Disentangler(args.n_concepts, args.noise_dim)
    m1 = InvarSennM1(conceptizer, parametrizer, aggregator)
    m2 = InvarSennM2(disentangler1, disentangler2)
    model = InvarSennTrainer(InvarSennM(m1, m2), robustness_reg = args.robustness_reg,recon_reg_concepts=args.recon_reg, dataset=args.dataset, path_pretrained=args.path_pretrained, disentangle_reg=args.disentangle_reg, update_ratio = args.update_ratio, disentangle_patience=args.disentangle_patience, batch_size=args.batch_size, lr_m1=args.lr_m1, lr_m2=args.lr_m2)
    model.train(args.n_epochs)
    
if __name__ == "__main__":
    main()
