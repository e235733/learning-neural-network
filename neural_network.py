import numpy as np

class NeuralNetworkModel:
    def __init__(self,explain,depend,eta):
        #説明変数X(d次元列ベクトルn個分), 目的変数y(n個分の0か1のラベル)
        self.X = explain
        self.N = self.X.shape[0]

        #各層のニューロンの数
        self.input_dim = self.X.shape[1]
        self.hidden_dim = 4
        self.output_dim = 2

        # Y は one_hot 表現にする
        self.Y = np.identity(self.output_dim)[depend]

        #調整すべきパラメータb:切片、w:d次元分の傾きを作成
        rng = np.random.default_rng()
        self.W1 = rng.standard_normal((self.input_dim, self.hidden_dim)) * 0.1
        self.W2 = rng.standard_normal((self.hidden_dim, self.output_dim)) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.b2 = np.zeros(self.output_dim)
        #bとwの学習率
        self.eta = eta

        self.loss_history = []

    
    def calc_loss(self):
        #損失評価
        logP = np.log(self.P)
        loss = -np.sum(self.Y * logP) / self.N
        print("loss: ", loss)
        self.loss_history.append(loss)

    def _sigmoid(self,Z):
        exp = np.exp(-Z)
        return 1 / (exp + 1)

    def calc_A1(self):
        Z1 = self.X @ self.W1 + self.b1
        self.A1 = self._sigmoid(Z1)

    def _softmax(self,Z):
        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        return exp_Z / np.sum(exp_Z,axis=1,keepdims=True)

    def calc_P(self):
        Z2 = self.A1 @ self.W2 + self.b2
        self.P = self._softmax(Z2)

    def grad(self):
        self.calc_A1()
        self.calc_P()

        da1 = (self.P - self.Y) @ self.W2.T / self.N
        dz1 = (self.A1 - self.A1*self.A1) * da1
        self.dw1 = self.X.T @ dz1
        self.db1 = np.sum(dz1,axis=0)

        dz2 = (self.P - self.Y) / self.N
        self.dw2 = self.A1.T @ dz2
        self.db2 = np.sum(dz2,axis=0)

    def shift(self):
        # 勾配を計算
        self.grad()

        # パラメータの更新
        self.W1 -= self.eta * self.dw1
        self.W2 -= self.eta * self.dw2
        self.b1 -= self.eta * self.db1
        self.b2 -= self.eta * self.db2

    def predict(self,X):
        Z1 = X @ self.W1 + self.b1
        A1 = self._sigmoid(Z1) 
        Z2 = A1 @ self.W2 + self.b2
        return self._softmax(Z2)            


    
if __name__ == "__main__":
    from xor_dataset import XorDataset

    ETA = 0.1

    data = XorDataset(10)
    print("data X, Y:")
    print(data.X)
    print(data.Y)
    model = NeuralNetworkModel(explain=data.X, depend=data.Y, eta=ETA)
    
    print(model.W1)
    model.shift()
    print(model.W1)
    
