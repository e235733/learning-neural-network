from xor_dataset import XorDataset
from sklearn_datasets import MoonsDataset, GaussianQuantilesDataset

from plotter import Plotter
from neural_network import NeuralNetworkModel
import function as fn

import numpy as np

def main():

    NUM_DATA = 250
    NUM_STEPS = 10000

    ACT_FUNCTION = fn.LeakyReLU()
    HIDDEN_LAYER = [8, 8, 8, 8]
    ETA = 0.1

    #チェック時やデバッグ時はTrue
    IS_DETAIL_MODE = False


    data = MoonsDataset(NUM_DATA)

    model = NeuralNetworkModel(data.X, data.Y, HIDDEN_LAYER, ETA, ACT_FUNCTION)

    plotter = Plotter(0.1, data.X, data.Y, IS_DETAIL_MODE)

    for step in range(NUM_STEPS):
        model.shift()
        model.calc_loss()
        if step % 400 == 0:
            plotter.show(model)
            loss = model.loss()
            print("loss: ", loss)
    plotter.show(model)

    # 正解率の計算
    Y_pred = np.argmax(model.P, axis=1)
    accuracy = np.mean(Y_pred == data.Y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    plotter.finish()

if __name__ == "__main__":
    main()
