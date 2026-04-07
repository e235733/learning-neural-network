import numpy as np
import function as fn

# ニューラルネットワークのモデルクラス
class NeuralNetworkModel:
    def __init__(self, input_dim, hidden_layer, output_dim, act_fn=fn.LeakyReLU(), output_fn=fn.Softmax(), eta=0.01, l2_lambda=0.005, alpha=0.9):
        
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

    def _initialize_parameters(self):
        self.W = []
        self.b = []
        
        rng = np.random.default_rng()
        
        layers = [self.input_dim] + self.hidden_layer + [self.output_dim]
        
        for i in range(len(layers) - 1):
            head = layers[i]
            tail = layers[i+1]
            
            # 活性化関数に基づいた初期化スケールを取得
            scale = self.act_fn.initialization(head, tail)
            
            w = rng.standard_normal((head, tail)) * scale
            b = np.zeros(tail)
            
            self.W.append(w)
            self.b.append(b)

        # Momentum 用の速度 V をゼロ初期化
        self.V_w = [np.zeros_like(w) for w in self.W]
        self.V_b = [np.zeros_like(b) for b in self.b]
    
    def upd_A_P(self, X):       
        # A の更新 
        fn_act = self.act_fn
        self.A = [X]
        for i in range(self.depth):
            Z = self.A[i] @ self.W[i] + self.b[i]
            self.A.append(fn_act.value(Z))
        # P の更新 
        o_fn = self.output_fn        
        L = self.depth
        Z = self.A[L] @ self.W[L] + self.b[L]
        self.P = o_fn.value(Z)        

    def upd_dW_db(self, A, dZ, threshold=5.0):        
        self.dW = []
        self.db = []
        L = self.depth
        for i in range(L+1):
            dw = A[i].T @ dZ[i]           
            dw += self.l2_lambda * self.W[i]
            
            # 勾配クリッピング
            dw_clipped = np.clip(dw, -threshold, threshold)
            
            self.dW.append(dw_clipped)
            db = np.sum(dZ[i], axis=0)
            self.db.append(db)

    def grad(self, X, Y):
        # 前向き伝播
        self.upd_A_P(X)
        dZ = []

        # 逆向き伝播
        dz = self.output_fn.dLoss(self.P, Y)
        dZ.append(dz)
        for i in range(self.depth, 0, -1):
            da = dz @ self.W[i].T
            diff = self.act_fn.diff(self.A[i])
            dz = diff * da
            dZ.append(dz)
        dZ.reverse()

        # 勾配の計算
        self.upd_dW_db(self.A, dZ)

    def upd_W(self, i):
        # 速度 V_w の更新: V = α*V - η*dW
        self.V_w[i] = self.alpha * self.V_w[i] - self.eta * self.dW[i]
        # 重み W の更新: W += V
        self.W[i] += self.V_w[i]
    
    def upd_b(self, i):
        # バイアス b の更新
        self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
        self.b[i] += self.V_b[i]

    def shift(self, X, Y):
        # 勾配を計算してパラメータを更新
        self.grad(X, Y)

        L = self.depth
        for i in range(L+1):
            self.upd_W(i)
            self.upd_b(i)

    def predict(self, X):
        A = X       
        for i in range(self.depth):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        i = self.depth
        Z = A @ self.W[i] + self.b[i]
        return self.output_fn.value(Z)
    
    def loss(self, X, Y):
        P = self.predict(X)
        o_fn = self.output_fn
        base_loss = o_fn.Loss(P, Y)
        
        l2_penalty = 0.0
        for w in self.W:
            l2_penalty += np.sum(w ** 2) 
        l2_penalty *= (self.l2_lambda / 2)

        return l2_penalty + base_loss

    
if __name__ == "__main__":
    import function as fn

    # 簡易テスト
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # XOR っぽいラベル

    model = NeuralNetworkModel(
        input_dim=2,
        hidden_layer=[4, 4],
        output_dim=2,
        act_fn=fn.Sigmoid(),
        output_fn=fn.Softmax(),
        eta=0.1
    )

    print("Initial Loss:", model.loss(X, Y))
    for _ in range(100):
        model.shift(X, Y)
    print("Loss after 100 steps:", model.loss(X, Y))
    print("Predictions:\n", model.predict(X))
