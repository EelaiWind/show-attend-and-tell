from __future__ import print_function
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.utils import load_word_to_idx
from core.split_data_loader import SplitDataLoader
import sys
import os

BATCH_SIZE = 128
MAX_EPOCH = 10

def main():
    if len(sys.argv) < 2:
        print("usage %s <log_root_path>" % sys.argv[0], file=sys.stderr)
        exit(1)

    log_root_path = sys.argv[1]
    if not os.path.exists(log_root_path):
        os.makedirs(log_root_path) 
    
    model_path = os.path.join(log_root_path, "model", "lstm")
    test_model = os.path.join(model_path, 'model-%d' % MAX_EPOCH)

    print("Test model : %s" % test_model)

    # load train dataset
    #train_datasets = SplitDataLoader(root_path="./data", dataset_directoris=[ ("train_%d" % i) for i in xrange(5) ], batch_size=BATCH_SIZE)
    word_to_idx = load_word_to_idx('./data/train/')
    
    # load val dataset to print out bleu scores every epoch
    test_data = load_coco_data(data_path='./data', split='test')

    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, None, test_data, test_model=test_model, batch_size=30)
    solver.test(test_data, '.', attention_visualization=True, save_sampled_captions=True)

if __name__ == "__main__":
    main()