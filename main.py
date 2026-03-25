from xor_dataset import XorDataset
from plotter import Plotter
from neural_network import NeuralNetworkModel

import numpy as np

def main():

    #普通の実行なら1,デバッグやチェック時は2を選択
    RUN_MODE = 2

    NUM_DATA = 250
    NUM_STEPS = 10000
    ETA = 2

    data = XorDataset(NUM_DATA)

    model = NeuralNetworkModel(data.X, data.Y,[6,6], ETA)

    plotter = Plotter(0.1, data.X, data.Y)

    for step in range(NUM_STEPS):
        model.shift()
        model.calc_loss()
        if step % (200 * RUN_MODE) == 0:
            plotter.show(model)
            if RUN_MODE == 2:
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
