import function as fn

class FlamePackage:
    def __init__(self, input_dim, output_dim, hidden_layer):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = hidden_layer

class FunctionPackage:
    def __init__(self, act_fn :fn.Function, output_fn :fn.OutputFunction):
        self.act = act_fn
        self.output = output_fn

class CoefficientPackage:
    def __init__(self, eta, l2_lambda, alpha):
        self.eta = eta
        self.l2_lambda = l2_lambda
        self.alpha = alpha