from mnist_dataset import MnistDataset
from plotter import Plotter
from neural_network import NeuralNetworkModel
import function as fn
import numpy as np
from sklearn.model_selection import train_test_split

def main():

    NUM_DATA = 20000
    NUM_STEPS = 1000

    ACT_FUNCTION = fn.LeakyReLU()
    HIDDEN_LAYER = [128, 32]
    ETA = 0.1

    #チェック時やデバッグ時はTrue
    IS_DETAIL_MODE = True

    # --- データの準備 ---
    all_data = MnistDataset(NUM_DATA)
    # データを訓練用(80%)とテスト用(20%)に分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_data.X, all_data.Y, test_size=0.2, random_state=42
    )

    # --- モデルとプロッターの初期化 ---
    OUTPUT_FUNCTION = fn.Softmax(len(X_train))
    fn_box = fn.FunctionBox(ACT_FUNCTION, OUTPUT_FUNCTION)
    
    model = NeuralNetworkModel(X_train, Y_train, HIDDEN_LAYER, ETA, fn_box)
    # プロッターには訓練データを渡して初期化
    plotter = Plotter(0.1, X_train, Y_train, IS_DETAIL_MODE)

    for step in range(NUM_STEPS):
        model.shift()
        model.calc_loss()
        # テストデータの損失を計算
        model.calc_val_loss(X_test, Y_test)
        
        if step % 500 == 0:
            plotter.show(model)
            loss = model.loss_history[-1]
            val_loss = model.val_loss_history[-1]
            print(f"Step {step}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            
    plotter.show(model)

    # 訓練データの正解率
    Y_train_pred = np.argmax(model.predict(X_train), axis=1)
    train_accuracy = np.mean(Y_train_pred == Y_train)
    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")

    # 正解率の計算（テストデータで最終確認）
    Y_test_pred = np.argmax(model.predict(X_test), axis=1)
    test_accuracy = np.mean(Y_test_pred == Y_test)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    plotter.finish()

if __name__ == "__main__":
    main()
