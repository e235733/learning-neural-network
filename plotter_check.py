import matplotlib.pyplot as plt
import numpy as np

class CheckPlotter:
    def __init__(self, interval, X, Y):
        self.interval = interval
        self.X = X
        self.Y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        
        # figsize=(幅, 高さ) を指定します。
        # 一般的なノートPCなら (12, 7) や (10, 6) あたりが収まりが良いです。
        self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 7))
        
        # グラフ同士のラベルが重ならないように自動調整
        plt.tight_layout()
        
        # 余白をもう少し微調整したい場合は subplots_adjust を併用
        plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)

    def show(self, model):
        # 1. 学習曲線 (左上)
        ax_loss = self.axs[0, 0]
        ax_loss.cla()
        ax_loss.plot(model.loss_history, color='purple', linewidth=2)
        ax_loss.set_title("Learning Curve")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")

        # 2. 決定境界 (中央上)
        ax_data = self.axs[0, 1]
        ax_data.cla()
        self._plot_2d(model, ax_data) # 第2引数にaxを渡すように修正

        # 3. 重み(W)の分布 (右上)
        ax_w = self.axs[0, 2]
        ax_w.cla()
        for i, w in enumerate(model.W):
            ax_w.hist(w.flatten(), bins=30, alpha=0.5, label=f"Layer {i}")
        ax_w.set_title("Weight Distribution (W)")
        ax_w.legend()

        # 4. 勾配(dW)の大きさ (右下)
        ax_dw = self.axs[1, 2]
        ax_dw.cla()
        if hasattr(model, 'dW'):
            # 各層の勾配の絶対値の平均を棒グラフで表示
            layer_labels = [f"L {i}" for i in range(len(model.dW))]
            grad_means = [np.mean(np.abs(dw)) for dw in model.dW]
            ax_dw.bar(layer_labels, grad_means, color='orange')
        ax_dw.set_title("Mean Gradient Magnitude (|dW|)")

        # 5. 空いている軸（左下・中央下）を非表示にする（任意）
        # self.axs[1, 0].axis('off')
        # self.axs[1, 1].axis('off')

        plt.pause(self.interval)

    def finish(self):
        plt.show()

    def _plot_2d(self, model, ax): # 第2引数 ax を追加
        ax.set_title("Decision Boundary")
        
        # 描画範囲の設定
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 予測
        probs = model.predict(grid_points)
        probs_class1 = probs[:, 1]
        predicted_grid = probs_class1.reshape(xx.shape)

        # 塗りつぶしと境界線
        ax.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='bwr', levels=np.linspace(0, 1, 11))
        ax.contour(xx, yy, predicted_grid, levels=[0.5], colors='green', linewidths=2)
        
        # 元データの散布図
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap='bwr', edgecolors='k')