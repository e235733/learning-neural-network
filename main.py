from xor_dataset import XorDataset
from plotter import Plotter
from neural_network import NeuralNetworkModel

def main():

    NUM_STEPS = 100

    data = XorDataset(100)

    model = NeuralNetworkModel()

    plotter = Plotter(0.1, data.X, data.Y)

    for step in range(NUM_STEPS):
        model.shift()
        plotter.show(model)

if __name__ == "__main__":
    main()
