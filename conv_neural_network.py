import numpy as np
from neural_network import ModelSetter

# 畳み込みニューラルネットワークのモデルクラス
class ConvolutionalNeuralNetwork:
    def __init__(self, model_setter: ModelSetter):
        if model_setter.is_not_flame_set or model_setter.is_not_function_set or model_setter.is_not_coefficient_set:
            raise ValueError("Model parameters not fully set. Please ensure all settings are configured before creating the model.")
        # モデルのパラメータを ModelSetter から受け取る
        self.W = model_setter.W
        self.b = model_setter.b
        self.V_W = model_setter.V_W
        self.V_b = model_setter.V_b
        self.dep = model_setter.dep
        self.act_fn = model_setter.act_fn # 隠れ層の活性化関数
        self.output_fn = model_setter.output_fn # 出力層の活性化関数
        self.eta = model_setter.eta # b と W の学習率
        self.l2_lambda = model_setter.l2_lambda # L2正則化のペナルティ
        self.alpha = model_setter.alpha # 慣性係数

    def forward(self, X):
        # フォワードパスの実装
        self.A = [X]
        for i in range(self.dep):
            Z = self.A[i] @ self.W[i] + self.b[i]
            self.A.append(self.act_fn.value(Z))
        Z = self.A[self.dep] @ self.W[self.dep] + self.b[self.dep]
        self.P = self.output_fn.value(Z)

    def backward(self, Y):
        # バックワードパスの実装
        self.dW = []
        self.db = []
        L = self.dep
        dZ = self.output_fn.derivative(self.P, Y)
        for i in range(L, -1, -1):
            dw = self.A[i].T @ dZ + self.l2_lambda * self.W[i]
            db = np.sum(dZ, axis=0)
            self.dW.append(dw)
            self.db.append(db)
            if i > 0:
                dA_prev = dZ @ self.W[i].T
                dZ = dA_prev * self.act_fn.derivative(self.A[i])
        self.dW.reverse()
        self.db.reverse()

    def update_parameters(self):
        # パラメータの更新
        for i in range(self.dep + 1):
            self.V_W[i] = self.alpha * self.V_W[i] - self.eta * self.dW[i]
            self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
            self.W[i] += self.V_W[i]
            self.b[i] += self.V_b[i]

    def predict(self, X):
        # 予測の実装
        A = X
        for i in range(self.dep):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        Z = A @ self.W[self.dep] + self.b[self.dep]
        P = self.output_fn.value(Z)
        return P

    def loss(self, X, Y):
        # 損失関数の実装（例: クロスエントロピー損失）
        self.forward(X)
        m = Y.shape[0]
        log_likelihood = -np.log(self.P[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def shift(self, X, Y):
        # 勾配を計算してパラメータを更新
        self.backward(Y)
        self.update_parameters()
    
    def train(self, X, Y, epochs):
        for _ in range(epochs):
            self.shift(X, Y)
    
    def evaluate_accuracy(self, X, Y: np.ndarray):
        # 予測と正解を比較して精度を計算
        predicted_classes = np.argmax(self.predict(X), axis=1)
        Y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        accuracy = np.mean(predicted_classes == Y_labels)
        return accuracy
    
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