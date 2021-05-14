import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

# locals
from SENN.models import Senn
from SENN.conceptizers import SENNConceptizer
from SENN.parametrizers import SENNParametrizer
from SENN.aggregators import BaseAggregator
from SENN.trainers import VanillaSennTrainer
from SENN.argparser import Argparser

parser = Argparser("senn")
args = parser.parse_args()

def main():
    DATASET = args.dataset
    conceptizer = SENNConceptizer(args.n_concepts, DATASET)
    aggregator = BaseAggregator()
    parametrizer = SENNParametrizer(args.n_concepts, DATASET)
    model = VanillaSennTrainer(Senn(conceptizer, parametrizer, aggregator), dataset=DATASET, robustness_reg=args.robustness_reg, concept_reg=args.recon_reg, sparsity_reg=args.sparsity_reg, batch_size = args.batch_size, lr=args.lr, path_pretrained=args.path_pretrained)
    model.train(args.n_epochs)
    
if __name__ == "__main__":
    main()