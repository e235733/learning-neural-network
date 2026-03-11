from xor_dataset import XorDataset
from plotter import Plotter
from neural_network import NeuralNetworkModel

import numpy as np

def main():

    NUM_STEPS = 2000
    ETA = 2

    data = XorDataset(250)

    model = NeuralNetworkModel(data.X, data.Y, ETA)

    plotter = Plotter(0.1, data.X, data.Y)

    for step in range(NUM_STEPS):
        model.shift()
        model.calc_loss()
        if step % 20 == 0:
            plotter.show(model)

    # 正解率の計算
    Y_pred = np.argmax(model.P, axis=1)
    accuracy = np.mean(Y_pred == data.Y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    plotter.finish()

if __name__ == "__main__":
    main()
