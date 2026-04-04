import numpy as np
import function as fn

class ModelSetter:
    def __init__(self, batch_size = None, x_data:np.ndarray = None, y_data: np.ndarray = None):
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        self.is_not_flame_set = True
        self.is_not_function_set = True
        self.is_not_coefficient_set = True

    def setting_Flame(self, hidden_layer, input_dim = None, output_dim = None):
        if input_dim is None:
            input_dim = self.x_data.shape[1]
        if output_dim is None:
            output_dim = self.y_data.shape[1]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = hidden_layer
        self.is_not_flame_set = False

    def setting_Function(self, act_fn:fn.Function, output_fn:fn.OutputFunction):
        if act_fn is None:
            act_fn = fn.LeakyReLU()
        if output_fn is None:
            output_fn = fn.Softmax(self.batch_size)

        self.act_fn = act_fn
        self.output_fn = output_fn
        self.is_not_function_set = False

    def setting_Coefficient(self, eta = 0.02, l2_lambda = 0.01, alpha = 0.9):
        self.eta = eta
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.is_not_coefficient_set = False

    def create_model(self):
        if self.is_not_flame_set:
            raise ValueError("Flame parameters not set. Please call setting_Flame() before creating the model.")
        if self.is_not_function_set:
            self.setting_Function()
        if self.is_not_coefficient_set:
            self.setting_Coefficient()
        
        return NeuralNetworkModel(self)
        


class NeuralNetworkModel:
    def para_generation(self,head,tail):
        rng = np.random.default_rng()
        scale = self.act_fn.initialization(head, tail)
        w = rng.standard_normal((head, tail)) * scale
        b = np.zeros(tail)
        self.W.append(w)
        self.b.append(b)

    def para_setting(self, hidden_layer):
        #調整すべきパラメータb:切片(1×K),W:傾き(D×K)を隠れ層＋出力層の深さ(L+1)だけ作成    
        self.W = []
        self.b = []
        head = self.input_dim
        for tail in hidden_layer:
            self.para_generation(head, tail)
            head = tail
        tail = self.output_dim
        self.para_generation(head, tail)

        #新しく Momentum 用の変数として速度 V (Velocity) を追加
        # W と b と同じ形状のゼロ配列（速度）を作成
        self.V_W = [np.zeros_like(w) for w in self.W]
        self.V_b = [np.zeros_like(b) for b in self.b]

    def __init__(self, model_setter: ModelSetter):
        #各層のニューロンの数
        self.input_dim = model_setter.input_dim
        self.output_dim = model_setter.output_dim        
        hidden_layer = model_setter.hidden_layer
        self.dep = len(hidden_layer)

        self.act_fn = model_setter.act_fn # 隠れ層の活性化関数
        self.output_fn = model_setter.output_fn # 出力層の活性化関数

        self.eta = model_setter.eta # b と W の学習率
        self.l2_lambda = model_setter.l2_lambda # L2正則化のペナルティ
        self.alpha = model_setter.alpha # 慣性係数

        self.para_setting(hidden_layer)

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
            dw += self.l2_lambda * self.W[i]
            
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
        base_loss = o_fn.Loss(P, Y)
        
        l2_penalty = 0.0
        for w in self.W:
            l2_penalty += np.sum(w ** 2) 
        l2_penalty *= (self.l2_lambda / 2)

        return l2_penalty + base_loss

    
if __name__ == "__main__":
    from xor_dataset import XorDataset
    import function as fn

    DATA_SIZE = 10

    ETA = 0.1
    L2_LAMBDA = 0.005

    HIDDEN_LAYER = [4, 4]

    ACT_FN = fn.Sigmoid()
    OUTPUT_FN = fn.Softmax(DATA_SIZE)

    data = XorDataset(10)
    print("data X, Y:")
    print(data.X)
    print(data.Y)

    setter = ModelSetter()
    setter.setting_Flame(2, 2, HIDDEN_LAYER)
    setter.setting_Function(ACT_FN, OUTPUT_FN)
    setter.setting_Coefficient(ETA, L2_LAMBDA, 0.9)
    model = setter.create_model()

    print(model.W)
    model.shift()
    print(model.W)
    print("model A")
    print(model.A)
