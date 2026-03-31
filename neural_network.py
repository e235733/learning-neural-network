import numpy as np
import function as fn

class NeuralNetworkModel:
    def norm(self,Z,mean,std):
        # Z を正規化 
        eps = 1e-8
        norm_Z = (Z - mean) / (std + eps)
        return norm_Z
    
    def para_generation(self,head,tail):
        rng = np.random.default_rng()
        scale = self.act_fn.initialization(head, tail)
        w = rng.standard_normal((head, tail)) * scale
        b = np.zeros(tail)
        self.W.append(w)
        self.b.append(b)

    def para_setting(self, fn_box:fn.FunctionBox, alpha = 0.9):
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
        self.alpha = alpha # 慣性係数

    def __init__(self, input_dim, output_dim, struct, eta, fn_box:fn.FunctionBox):
        #各層のニューロンの数
        self.input_dim = input_dim
        self.hidden_dim = struct
        self.output_dim = output_dim
        self.dep = len(struct)

        self.para_setting(fn_box)

        # b と W の学習率
        self.eta = eta
        #グラフ作成用の損失記録
        self.train_loss_history = [] 
        self.test_loss_history = []

    
    def upd_A_P(self, X):       
        # A の更新
        fn = self.act_fn
        self.A = [X]
        for i in range(self.dep):
            Z = self.A[i] @ self.W[i] + self.b[i]
            self.A.append(fn.value(Z))
        # P の更新
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


    def grad(self, X, Y):
        self.upd_A_P(X)
        dZ = []

        dz = self.output_fn.dLoss(self.P, Y)
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

    def shift(self, X, Y):
        # 勾配を計算
        self.grad(X, Y)

        # パラメータの更新
        L = self.dep
        W = []
        b = []
        for i in range(L+1):
            W.append(self.new_W(i))
            b.append(self.new_b(i))
        self.W = W
        self.b = b


    def predict(self, X):
        A = X       
        for i in range(self.dep):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        i = self.dep
        Z = A @ self.W[i] + self.b[i]
        return self.output_fn.value(Z)
    
    def loss(self, X, Y):
        P = self.predict(X)
        o_fn = self.output_fn
        return o_fn.Loss(P, Y)

    
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
