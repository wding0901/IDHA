import argparse
import numpy as np
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed

from model import LightSAGE
from graphSAGE import GraphSAGE
from pinSAGE import PinSAGE
from ngcf import NGCF
from lightGCN import LightGCN

from trainer import MyTrainer
from recbole.quick_start import run_recbole
from recbole.trainer import Trainer


class_dict = {
    'LightSAGE': LightSAGE,
    'GraphSAGE': GraphSAGE,
    'PinSAGE': PinSAGE,
    'ngcf': NGCF,
    'lightGCN': LightGCN
}


def get_model(model_name):
    if model_name not in class_dict:
        raise ValueError("`model_name` [{}] is not the name of an existing model.".format(model_name))
    return class_dict[model_name]


def run_baseline(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    run_recbole(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict, saved=saved)


def run(args):
    model_class = get_model(args.model)
    # configurations initialization
    config = Config(model=model_class, dataset=args.dataset, config_file_list=args.config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset, logger).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    if args.model in ['ngcf', 'lightGCN']:
        trainer = Trainer(config, model)
    else:
        trainer = MyTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))


def test(args):
    model_class = get_model(args.model)
    # configurations initialization
    config = Config(model=model_class, dataset=args.dataset, config_file_list=args.config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print(train_data.dataset.num(config["USER_ID_FIELD"]))
    print(valid_data.dataset.num(config["USER_ID_FIELD"]))
    print(test_data.dataset.num(config["USER_ID_FIELD"]))

    inter = dataset.inter_matrix(form='coo').astype(np.float32)
    print(inter)
    inter = inter.tolil()
    train_size = train_data.dataset.num(config["USER_ID_FIELD"])
    random_row = np.random.choice(train_size, size=(train_size // 5))
    inter[random_row, :] = 0
    print(inter.tocoo())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help='The datasets can be: ml-1m, yelp, amazon-books, gowalla.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--model', type=str, default='LightSAGE',
                        help='The models can be: LightSAGE, NGCF, BPR, NeuMF.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = ['args/default.yaml']
    if args.config is not '':
        args.config_file_list.append(args.config)

    if args.dataset in ['amazon-electronics', 'amazon-cds', 'amazon-movies', 'gowalla', 'yelp', 'amazon-books']:
        args.config_file_list.append(f'args/{args.dataset}.yaml')

    # test(args)

    if args.model in class_dict:
        run(args)
    else:
        run_baseline(model=args.model, dataset=args.dataset, config_file_list=args.config_file_list)
