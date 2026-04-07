import numpy as np
import function as fn
from abc import ABC, abstractmethod

# ニューラルネットワークの抽象クラス
class Model(ABC):
    def __init__(self, input_dim, hidden_layer, output_dim,
                act_fn:fn.ActivationFunction = fn.LeakyReLU(),
                output_fn:fn.OutputFunction = fn.Softmax(),
                eta=0.01, l2_lambda=0.005, alpha=0.9):
        
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.output_dim = output_dim
        self.depth = len(hidden_layer)

        self.act_fn = act_fn # 隠れ層の活性化関数
        self.output_fn = output_fn # 出力層の活性化関数

        self.eta = eta # 学習率
        self.l2_lambda = l2_lambda # L2正則化のペナルティ
        self.alpha = alpha # 慣性係数

        # パラメータの初期化
        self._initialize_parameters()

        # グラフ作成用の損失記録
        self.train_loss_history = [] 
        self.test_loss_history = []

    @abstractmethod
    def _initialize_parameters(self):
        pass
    
    @abstractmethod
    def calc_forward_propagation(self, X):
        pass

    @abstractmethod
    def calc_backward_propagation(self, Y):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def shift(self, X, Y):
        # 勾配を計算
        self.calc_forward_propagation(X)
        self.calc_backward_propagation(Y)
        # パラメータを更新
        self.update_parameters()
    
    def loss(self, X, Y):
        P = self.predict(X)
        base_loss = self.output_fn.Loss(P, Y)
        
        l2_penalty = 0.0
        for w in self.W:
            l2_penalty += np.sum(w ** 2) 
        l2_penalty *= (self.l2_lambda / 2)

        return l2_penalty + base_loss
    
    def log_train_loss(self, X_train, Y_train):
        train_loss = self.loss(X_train, Y_train)
        self.train_loss_history.append(train_loss)
        return train_loss

    def log_test_loss(self, X_test, Y_test):
        test_loss = self.loss(X_test, Y_test)
        self.test_loss_history.append(test_loss)
        return test_loss

    def evaluate_accuracy(self, X, Y: np.ndarray):
        # 予測と正解を比較して精度を計算
        predicted_classes = np.argmax(self.predict(X), axis=1)
        Y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        accuracy = np.mean(predicted_classes == Y_labels)
        return accuracy
    
    def train(self, X, Y, epochs):
        for _ in range(epochs):
            self.shift(X, Y)
    
    def save_model(self, file_path):
        # モデルの保存（例: パラメータをファイルに保存）
        np.savez(file_path, W=self.W, b=self.b, V_W=self.V_W, V_b=self.V_b)
    
    def load_model(self, file_path):
        # モデルの読み込み（例: ファイルからパラメータを読み込む）
        data = np.load(file_path)
        self.W = data['W']
        self.b = data['b']
        self.V_W = data['V_W']
        self.V_b = data['V_b']