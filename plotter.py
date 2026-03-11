import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, interval, X, Y):
        self.interval = interval
        self.X = X
        self.Y = Y
        
        self.dim = 2
        self.fig = plt.figure(figsize=(12, 6)) 
        
        # 左は損失関数、右は分類結果
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_data = self.fig.add_subplot(1, 2, 2)
        
    def show(self, model):
        self.ax_data.cla()
        self._plot_2d(model)
                
        self.ax_loss.cla()
        self.ax_loss.plot(model.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title("Learning Curve")
        self.ax_loss.set_xlabel("Iteration (Step)")
        self.ax_loss.set_ylabel("Cross Entropy Loss")
        self.ax_loss.grid(True)
        
        plt.pause(self.interval)

    def finish(self):
        plt.show()

    def _plot_2d(self, model):
        self.ax_data.set_title("2D: Non-linear Decision Boundary")
        
        xx, yy = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        probs = model.predict(grid_points)
        
        probs_class1 = probs[:, 1]
        
        predicted_grid = probs_class1.reshape(xx.shape)

        self.ax_data.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='bwr', levels=np.linspace(0, 1, 11))
        
        self.ax_data.contour(xx, yy, predicted_grid, levels=[0.5], colors='green', linewidths=2)
        
        self.ax_data.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='bwr', edgecolors='k')