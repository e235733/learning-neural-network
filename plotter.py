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

    def _plot_2d(self, model):
        self.ax_data.set_title("Decision Boundary")
        x_min, x_max = np.min(self.X[:, 0])-1, np.max(self.X[:, 0])+1
        y_min, y_max = np.min(self.X[:, 1])-1, np.max(self.X[:, 1])+1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 128), np.linspace(y_min, y_max, 128))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        probs = model.predict(grid_points) 
        probs = probs.reshape(xx.shape)

        # 50% 境界線
        self.ax_data.contourf(xx, yy, probs, alpha=0.3, cmap='bwr', levels=np.linspace(0, 1, 11))
        self.ax_data.contour(xx, yy, probs, levels=[0.5], colors='green', linewidths=2)
        
        self.ax_data.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='bwr', edgecolors='k')