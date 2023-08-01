import os
import argparse
import pathlib
import numpy as np

import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator

tags = ["Accuracy", "Average_Precision", "F1_Score", "Loss"]
types = ["Test", "Train", "Validation"]
colors = ["b", "r", "g"]

def main(args):
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    All_files  = []
    tag_index = 0
    type_index = 0
    for root, dirs, files in os.walk(args.path):
        for file in files:
            All_files.append(os.path.join(root, file))

    """check the tags and types order"""
    for tag in tags:
        for type in types:
            print(tag + "_" + type)

    for name in sorted(All_files, key=str.lower):
        print(name)
        event_acc = event_accumulator.EventAccumulator(name, size_guidance={'images': 0})
        event_acc.Reload()

        scalars = {}
        if(len(event_acc.Tags()['scalars']) == 1):
            events = event_acc.Scalars(event_acc.Tags()['scalars'][0])
            scalars[event_acc.Tags()['scalars'][0]] = [event.value for event in events]
            plt.plot(range(1, len(scalars[event_acc.Tags()['scalars'][0]]) + 1),
                     scalars[event_acc.Tags()['scalars'][0]], label= types[type_index], color= colors[type_index])
            
            type_index += 1

        if(type_index > 2):
            plt.xlabel('epoch', fontsize=15)
            plt.ylabel(tags[tag_index], fontsize=15)
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1,2,0]
            plt.grid()
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=15)
            plt.title(args.path)
            plt.savefig(os.path.join(outdir, tags[tag_index] + ".png"))

            type_index = 0
            tag_index += 1
            plt.clf() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    main(args)