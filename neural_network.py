import numpy as np
import function as fn

class NeuralNetworkModel:
    def norm(self,Z,mean,std):
        # Z を正規化
        norm_Z = (Z - mean) / std
        return norm_Z
    
    def para_generation(self,head,tail):
        rng = np.random.default_rng()
        scale = np.sqrt(2 / (head + tail))
        w = rng.standard_normal((head, tail)) * scale
        b = np.zeros(tail)
        self.W.append(w)
        self.b.append(b)

    def __init__(self,explain,depend,struct,eta):
        #説明変数X(D×N), 目的変数Y(K×N)
        self.X = explain
        self.N = self.X.shape[0]

        # X を列ごとに正規化
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)
        self.norm_X = self.norm(self.X, self.X_mean, self.X_std)

        #各層のニューロンの数
        self.input_dim = self.X.shape[1]
        self.hidden_dim = struct
        self.output_dim = 2
        self.dep = len(struct)

        # Y は one_hot 表現にする
        self.Y = np.identity(self.output_dim)[depend]

        #調整すべきパラメータb:切片(1×K),W:傾き(D×K)を隠れ層＋出力層の深さ(L+1)だけ作成       
        self.W = []
        self.b = []
        head = self.input_dim
        for tail in self.hidden_dim:
            self.para_generation(head, tail)
            head = tail
        tail = self.output_dim
        self.para_generation(head, tail)

        #bとWの学習率
        self.eta = eta
        #グラフ作成用の損失記録
        self.loss_history = []

    def loss(self):
        logP = np.log(self.P)
        loss = -np.sum(self.Y * logP) / self.N
        return loss    
 
    def calc_loss(self):
        #損失評価
        loss = self.loss()
        self.loss_history.append(loss)

    def _sigmoid(self,Z):
        exp = np.exp(-Z)
        return 1 / (exp + 1)
    
    def _softmax(self,Z):
        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        return exp_Z / np.sum(exp_Z,axis=1,keepdims=True)

    
    def calc_A(self):
        # Aは勾配計算に使うだけなので保存せずに出力
        act_fn = fn.Sigmoid()
        A = [self.norm_X]
        for i in range(self.dep):
            Z = A[i] @ self.W[i] + self.b[i]
            A.append(act_fn.value(Z))
        return A

    def upd_P(self, A):
        # Pは損失評価にも使うのでインスタンス変数として保存
        L = self.dep
        Z = A[L] @ self.W[L] + self.b[L]
        self.P = self._softmax(Z)
    

    def upd_dW_db(self, A, dZ):
        _dW = []
        _db = []
        L = self.dep
        for i in range(L+1):
            dw = A[i].T @ dZ[i]
            _dW.append(dw)
            db = np.sum(dZ[i], axis=0)
            _db.append(db)
        self.dW = _dW
        self.db = _db


    def grad(self):
        A = self.calc_A()
        self.upd_P(A)
        dZ = []

        dz = (self.P - self.Y) / self.N
        dZ.append(dz)
        for i in range(self.dep, 0, -1):
            da = dz @ self.W[i].T
            dz = (A[i] - A[i]**2) * da
            dZ.append(dz)
        dZ.reverse()

        self.upd_dW_db(A, dZ)

    def new_W(self, i):
        return self.W[i] - self.eta * self.dW[i]
    
    def new_b(self, i):
        return self.b[i] - self.eta * self.db[i]

    def shift(self):
        # 勾配を計算
        self.grad()

        # パラメータの更新
        L = self.dep
        W = []
        b = []
        for i in range(L+1):
            W.append(self.new_W(i))
            b.append(self.new_b(i))
        self.W = W
        self.b = b


    def predict(self,X):
        A = self.norm(X, self.X_mean, self.X_std)       
        for i in range(self.dep):
            Z = A @ self.W[i] + self.b[i]
            A = self._sigmoid(Z)
        i = self.dep
        Z = A @ self.W[i] + self.b[i]
        return self._softmax(Z)


    
if __name__ == "__main__":
    from xor_dataset import XorDataset

    ETA = 0.1

    data = XorDataset(10)
    print("data X, Y:")
    print(data.X)
    print(data.Y)
    model = NeuralNetworkModel(explain=data.X, depend=data.Y, eta=ETA)
    
    print(model.W)
    model.shift()
    print(model.W)
