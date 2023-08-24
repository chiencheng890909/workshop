import os
import sys
from datetime import datetime
import random
import argparse

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np

from model.vertical_fl.OnePartyModel import OnePartyModel
from preprocess.credit_card import load_credit_card

now_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
os.chdir(sys.path[0] + "/../")  # change working directory
root = "data/credit_card/2cf/"
dataset = root + "A_credit_card.csv"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--leak-p', type=float, default=1.0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
X, y = load_credit_card(dataset)
print("X got {} dimensions".format(X.shape[1]))
name = "credit_card"

k_folds = 5
metric = ['auc', 'f1_score', 'average_precision', 'accuracy']
results = []

for index in range(k_folds):
    model = OnePartyModel(model_name=name + "_" + now_string,
                        task='binary_cls',
                        metrics=metric,
                        n_classes=2,
                        val_rate=0.1,
                        test_rate=0.2,
                        device='cuda:{}'.format(args.gpu),
                        hidden_sizes=[400, 200],
                        train_batch_size=4096,
                        test_batch_size=4096,
                        num_epochs=50,
                        learning_rate=3e-3,
                        weight_decay=1e-5,
                        num_workers=4 if sys.gettrace() is None else 0,
                        use_scheduler=False,
                        sche_factor=0.1,
                        sche_patience=10,
                        sche_threshold=0.0001,
                        writer_path="runs/{}_{}".format(name, now_string),
                        model_save_path="ckp/{}_{}.pth".format(name, now_string)
                        )

    result = model.train_single(X, y, scale=True, class_weight=True, seed=random.randint(0, 1000))
    results.append(result)
results = np.array(results)
print(results.shape)

for index in range(k_folds):
    print("k_fold", str(index + 1),":")
    for i in range(len(metric)):
        print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
            .format(metric[i], results[index, 0, i],
                    results[index, 1, i], results[index, 2, i]))
print("Mean:")
for i in range(len(metric)):
    print("          {:<17s}: Train {:.4f}, Val {:.4f}, Test {:.4f}"
        .format(metric[i], np.mean(results[:, 0, i]),
                np.mean(results[:, 1, i]), np.mean(results[:, 2, i])))
