
from utils import tab_printer, Logger
from CasSeqGCN import CasSeqGCNTrainer
from param_parser import parameter_parser
import sys
import time


def main():
    start = time.time()
    args = parameter_parser()
    sys.stdout = Logger(args.result_log)
    tab_printer(args)
    print('learning_rate: ', args.learning_rate)
    print('weight_decay: ', args.weight_decay)
    model = CasSeqGCNTrainer(args)
    model.fit()
    # model.test()
    end = time.time()
    print('consume time: ', (end-start) / 60, ' minutes\n')

if __name__ == '__main__':
    main()