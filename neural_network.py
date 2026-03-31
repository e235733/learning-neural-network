import numpy as np
import function as fn

class NeuralNetworkModel:
    def norm(self,Z,mean,std):
        # Z を正規化
        norm_Z = (Z - mean) / std
        return norm_Z
    
    def para_generation(self,head,tail):
        rng = np.random.default_rng()
        scale = self.act_fn.initialization(head, tail)
        w = rng.standard_normal((head, tail)) * scale
        b = np.zeros(tail)
        self.W.append(w)
        self.b.append(b)

    def __init__(self, explain:np, depend:np, struct, eta, fn_box:fn.FunctionBox):
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
        self.output_dim = int(np.max(depend) + 1)
        self.dep = len(struct)

        # Y は one_hot 表現にする
        self.Y = np.identity(self.output_dim)[depend]

        #調整すべきパラメータb:切片(1×K),W:傾き(D×K)を隠れ層＋出力層の深さ(L+1)だけ作成
        self.act_fn = fn_box.act
        self.output_fn = fn_box.output     
        self.W = []
        self.b = []
        head = self.input_dim
        for tail in self.hidden_dim:
            self.para_generation(head, tail)
            head = tail
        tail = self.output_dim
        self.para_generation(head, tail)

        #新しく Momentum 用の変数として速度 V (Velocity) を追加
        # W と b と同じ形状のゼロ配列（速度）を作成
        self.V_W = [np.zeros_like(w) for w in self.W]
        self.V_b = [np.zeros_like(b) for b in self.b]
        self.alpha = 0.5 # 慣性係数（過去の勢いを50%引き継ぐ）

        # b と W の学習率
        self.eta = eta
        #グラフ作成用の損失記録
        self.loss_history = [] 

    def loss(self):
        o_fn = self.output_fn
        return o_fn.Loss(self.P, self.Y)
 
    def calc_loss(self):
        #損失評価
        loss = self.loss()
        self.loss_history.append(loss)

    
    def upd_A(self):
        # Aも確認可能なようにインスタンス変数に保存
        fn = self.act_fn
        self.A = [self.norm_X]
        for i in range(self.dep):
            Z = self.A[i] @ self.W[i] + self.b[i]
            self.A.append(fn.value(Z))

    def upd_P(self):
        # Pは損失評価にも使うのでインスタンス変数として保存
        o_fn = self.output_fn
        L = self.dep
        Z = self.A[L] @ self.W[L] + self.b[L]
        self.P = o_fn.value(Z)

    def upd_dW_db(self, A, dZ, threshold = 1.0):        
        _dW = []
        _db = []
        L = self.dep
        for i in range(L+1):
            dw = A[i].T @ dZ[i]
            
            # --- 勾配クリッピングを追加 ---
            # dw が -1.0 ～ 1.0 の範囲に収まるように制限
            dw_clipped = np.clip(dw, -threshold, threshold)
            
            _dW.append(dw_clipped)
            db = np.sum(dZ[i], axis=0)
            _db.append(db)
        self.dW = _dW
        self.db = _db


    def grad(self):
        self.upd_A()
        self.upd_P()
        dZ = []

        dz = self.output_fn.dLoss(self.P, self.Y)
        dZ.append(dz)
        for i in range(self.dep, 0, -1):
            da = dz @ self.W[i].T
            diff = self.act_fn.diff(self.A[i])
            dz = diff * da
            dZ.append(dz)
        dZ.reverse()

        self.upd_dW_db(self.A, dZ)

    def new_W(self, i):
        # 速度 V_W の更新: V = α*V - η*dW
        self.V_W[i] = self.alpha * self.V_W[i] - self.eta * self.dW[i]
        # 重み W の更新: W = W + V
        _new_W = self.W[i] + self.V_W[i]
        return _new_W
    
    def new_b(self, i):
        # バイアス b も同様に速度 V_b を更新
        self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
        _new_b = self.b[i] + self.V_b[i]
        return _new_b

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
            A = self.act_fn.value(Z)
        i = self.dep
        Z = A @ self.W[i] + self.b[i]
        return self.output_fn.value(Z)


    
if __name__ == "__main__":
    from xor_dataset import XorDataset
    import function as fn

    ETA = 0.1
    STRUCT = [4, 4]

    ACT_FN = fn.Sigmoid()

    data = XorDataset(10)
    print("data X, Y:")
    print(data.X)
    print(data.Y)
    model = NeuralNetworkModel(explain=data.X, depend=data.Y, struct=STRUCT, eta=ETA, act_fn=ACT_FN)

    print(model.W)
    model.shift()
    print(model.W)
    print("model A")
    print(model.A)
