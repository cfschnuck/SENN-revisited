import sys
from os.path import dirname, realpath
import logging
sys.path.append(dirname(dirname(realpath(__file__))))

# Local imports
from SENN.models import GaussSiamSenn
from SENN.simsiam import GaussTripletSimSiam
from SENN.parametrizers import SENNParametrizer
from SENN.aggregators import BaseAggregator
from SENN.trainers import GaussTripletSiamSennTrainer
from SENN.argparser import Argparser

parser = Argparser("vsiamsenn")
args = parser.parse_args()

def main():
    DATASET = args.dataset
    n_channels = 3 if DATASET == "CIFAR10" else 1
    conceptizer = GaussTripletSimSiam(args.n_concepts, n_channels)
    aggregator = BaseAggregator()
    parametrizer = SENNParametrizer(args.n_concepts, DATASET)
    model = GaussTripletSiamSennTrainer(GaussSiamSenn(conceptizer, parametrizer, aggregator), dataset=DATASET, robustness_reg=args.robustness_reg, batch_size = args.batch_size, simsiam_reg=args.simsiam_reg, beta_reg = args.beta_reg, rob_reg_concepts = args.robustness_reg_concepts, path_pretrained=args.path_pretrained)
    model.train(args.n_epochs)

if __name__ == '__main__':
    main()