import argparse
import wandb


def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--train', '-t', default=0, action='store_true')
    parser.add_argument('--oTest', '-oTest', default=0, action='store_true')

    parser.add_argument('--tsne', '-tsne', default=0, action='store_true')
    parser.add_argument('--cluster', '-cluster', default=0, action='store_true')
    parser.add_argument('--arcface', '-arcface', default=0, action='store_true')
    parser.add_argument('--deepclassifier', '-deepclassifier', default=0, action='store_true')
    parser.add_argument('--kl', '-kl', default=0, action='store_true')
    parser.add_argument('--falselabels', '-falselabels', default=0, action='store_true')
    parser.add_argument('--epochTSNE', default=198, type=int, help='Classes used in testing')
    parser.add_argument('--stop', default=1, type=int, help='Classes used in testing')

    parser.add_argument('--NN', '-NN', default=0, action='store_true')
    parser.add_argument('--overclass', '-overclass', default=0, action='store_true')
    parser.add_argument('--imbalance', '-imbalance', default=0, action='store_true')
    parser.add_argument('--openset', '-openset', default=0, action='store_true')
    parser.add_argument('--union', '-union', default=0, action='store_true')
    parser.add_argument('--extraclass', default=2, type=int, help='Classes used in testing')
    parser.add_argument('--featperclass', '-featperclass', default=0, action='store_true')
    parser.add_argument('--level', '-level', default=0, action='store_true')
    parser.add_argument('--deepNN', '-deepNN', default=0, action='store_true')
    parser.add_argument('--reserve', '-reserve', default=0, action='store_true')
    parser.add_argument('--extraLayer', '-extraLayer', default=0, action='store_true')
    parser.add_argument('--part', '-part', default=0, action='store_true')
    parser.add_argument('--perepo', '-perepo', default=0, action='store_true')
    parser.add_argument('--osvm', '-osvm', default=0, action='store_true')
    parser.add_argument('--orth', '-orth', default=0, action='store_true')
    parser.add_argument('--trans', '-trans', default=0, action='store_true')
    parser.add_argument('--f', '-f', default=0, action='store_true')
    parser.add_argument('--unsave', '-unsave', default=0, action='store_true')
    parser.add_argument('--pca', '-pca', default=0, action='store_true')
    parser.add_argument('--pool', '-pool', default=0, action='store_true')
    parser.add_argument('--numOftrain', default=20, type=int, help='Classes used in testing')
    parser.add_argument('--Kunif', default=2, type=float, help='Classes used in testing')
    parser.add_argument('--Kuniq', default=0.1, type=float, help='Classes used in testing')
    parser.add_argument('--mu', default=0.1, type=float, help='Classes used in testing')
    parser.add_argument('--Nclasses', default=6, type=float, help='Classes used in testing')
    parser.add_argument('--fc', '-fc', default=0, action='store_true')
    parser.add_argument('--falseLayer', default=0, type=float, help='Classes used in testing')
    parser.add_argument('--entropy', '-entropy', default=0, action='store_true')
    parser.add_argument('--lamda', default=5e-4, type=float, help='Classes used in testing')
    parser.add_argument('--batch', default=500, type=int, help='Classes used in testing')
    parser.add_argument('--load_path', default="", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--save_path', default="", type=str,
                        help="Path to save the ensemble weights")
    parser.add_argument('--rand', '-rand', default=0, action='store_true')
    parser.add_argument('--mnist', '-mnist', default=0, action='store_true')
    parser.add_argument('--ph1', '-ph1', default=0, action='store_true')
    parser.add_argument('--ph2', '-ph2', default=0, action='store_true')
    parser.add_argument('--constLayer', '-constLayer', default=0, action='store_true')
    parser.add_argument('--time', '-time', default=0, action='store_true')

    parser.add_argument('--ae', '-ae', default=0, action='store_true')

    args = parser.parse_args()
    return args


def configWand(args):
    if args.trans:
        opt = 'adam'
    else:
        opt = 'SGD'
    config_defaults = {
        'epochs': 200,
        'batch_size': args.batch,
        'learning_rate': 1e-3,
        'optimizer': opt,
        'overclass': args.overclass,
        'openset': args.openset,
        'union': args.union,
        'Kunif': args.Kunif,
        'Kuniq': args.Kuniq,
        'featperclass': args.featperclass,
        'trans': args.trans,
        'lr': args.lr,

    }
    return config_defaults
