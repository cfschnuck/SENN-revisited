import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

# Local imports
from SENN.models import VAESenn
from SENN.conceptizers import VAEConceptizer
from SENN.parametrizers import SENNParametrizer
from SENN.aggregators import BaseAggregator
from SENN.trainers import VAETrainerSeperated
from SENN.argparser import Argparser

parser = Argparser("vaesenn")
args = parser.parse_args()

def main():
    DATASET = args.dataset
    conceptizer = VAEConceptizer(args.n_concepts, args.n_styles, dataset=DATASET)
    aggregator = BaseAggregator()
    parametrizer = SENNParametrizer(args.n_concepts, DATASET)
    model = VAETrainerSeperated(VAESenn(conceptizer, parametrizer, aggregator), dataset=DATASET, robustness_reg=args.robustness_reg, batch_size = args.batch_size, lr=args.lr, beta_reg_styles=args.beta_reg_styles, beta_reg_concepts=args.beta_reg_concepts, path_pretrained=args.path_pretrained)
    model.train(args.n_epochs_pretrain, args.n_epochs)

if __name__ == '__main__':
    main()