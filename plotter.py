import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, interval, X, Y, is_detail_mode=False):
        self.interval = interval
        self.X = X
        # one-hot to integer labels
        self.Y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        self.num_classes = int(np.max(self.Y_labels) + 1)
        self.is_detail_mode = is_detail_mode
        self.input_dim = X.shape[1]
        
        if self.is_detail_mode:
            # Detail mode: 2x3 grid.
            self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 7))
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
            self.ax_loss = self.axs[0, 0]
            self.ax_data = self.axs[0, 1]
            if self.input_dim not in [1, 2]:
                self.ax_data.set_axis_off()
                self.ax_data.set_title(f"No plot ({self.input_dim}D)")
        else:
            # Default mode: 1x2 if 1D or 2D, 1x1 if not
            if self.input_dim in [1, 2]:
                self.fig = plt.figure(figsize=(12, 6))
                self.ax_loss = self.fig.add_subplot(1, 2, 1)
                self.ax_data = self.fig.add_subplot(1, 2, 2)
            else:
                self.fig = plt.figure(figsize=(6, 6))
                self.ax_loss = self.fig.add_subplot(1, 1, 1)
                self.ax_data = None

    def show(self, model):
        self._show_loss(model)
        
        if self.input_dim == 2:
            self.ax_data.cla()
            self._plot_2d(model, self.ax_data)
        elif self.input_dim == 1:
            self.ax_data.cla()
            self._plot_1d(model, self.ax_data)
            
        if self.is_detail_mode:
            self._show_network_stats(model)
        
        plt.pause(self.interval)

    def _show_loss(self, model):
        self.ax_loss.cla()
        self.ax_loss.plot(model.train_loss_history, color='purple', linewidth=2, label='Train Loss')
        if hasattr(model, 'test_loss_history') and len(model.test_loss_history) > 0:
            self.ax_loss.plot(model.test_loss_history, color='orange', linewidth=2, label='Test Loss')
        self.ax_loss.set_title("Learning Curve")
        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

    def _show_network_stats(self, model):
        ax_w = self.axs[1, 0]
        ax_w.cla()
        for i, w in enumerate(model.W):
            ax_w.hist(w.flatten(), bins=30, alpha=0.5, label=f"Layer {i}")
        ax_w.set_title("Weight Distribution (W)")
        ax_w.legend()

        ax_A = self.axs[1, 1]
        ax_A.cla()
        for i, a in enumerate(model.A):
            if i == 0:
                continue
            ax_A.hist(a.flatten(), bins=30, alpha=0.5, label=f"Layer {i}")
        ax_A.set_title("Activation Distribution (A)")
        ax_A.legend()

        ax_dw = self.axs[1, 2]
        ax_dw.cla()
        if hasattr(model, 'dW'):
            layer_labels = [f"L {i}" for i in range(len(model.dW))]
            grad_means = [np.mean(np.abs(dw)) for dw in model.dW]
            ax_dw.bar(layer_labels, grad_means, color='orange')
        ax_dw.set_title("Mean Gradient Magnitude (|dW|)")

    def finish(self):
        plt.show()

    def _plot_1d(self, model, ax):
        ax.set_title(f"Decision Boundary (1D, {self.num_classes} classes)")
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        probs = model.predict(xx)
        
        cmap = plt.get_cmap('tab10')
        for c in range(self.num_classes):
            ax.plot(xx, probs[:, c], color=cmap(c), label=f'Prob Class {c}')
            
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        
        ax.scatter(self.X[:, 0], np.zeros_like(self.Y_labels), c=self.Y_labels, cmap='tab10', edgecolors='k', zorder=3, label='True Labels')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Input X")
        ax.set_ylabel("Probability")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)

    def _plot_2d(self, model, ax):
        ax.set_title(f"Decision Boundary ({self.num_classes} classes)")
        
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        probs = model.predict(grid_points)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_grid = predicted_classes.reshape(xx.shape)

        ax.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='tab10', levels=np.arange(self.num_classes + 1) - 0.5)
        
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.Y_labels, cmap='tab10', edgecolors='k')
