from xor_dataset import XorDataset
from sklearn_datasets import MoonsDataset, GaussianQuantilesDataset
from mnist_dataset import MnistDataset
from plotter import Plotter
from data_loader import DataLoader, DataNormalizer
from neural_network import ModelSetter
import function as fn
import numpy as np
from sklearn.model_selection import train_test_split

import time

def main():

    NUM_DATA = 60000
    NUM_EPOCHS = 200
    BATCH_SIZE = 512

    # setting_Flame に  登録するパラメーター
    HIDDEN_LAYER = [128, 128, 64, 64, 32, 32]
    
    # setting_Function に登録する活性化関数
    ACT_FUNCTION = fn.LeakyReLU()
    OUTPUT_FUNCTION = fn.Softmax()

    # setting_Coefficient に登録するパラメーター
    ETA = 0.01  # 学習率
    
    L2_LAMBDA = 0.005  # L2正則化のペナルティ
    ALPHA = 0.9  # Momentum の慣性係数

    #チェック時やデバッグ時はTrue
    IS_DETAIL_MODE = True

    # --- データの準備 ---
    all_data = MnistDataset(NUM_DATA)

    # テストデータのサイズを制限（最大1000サンプル or 20%）
    test_data_size = min(int(0.2 * NUM_DATA), 1000)
    # データを訓練用とテスト用に分割
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_data.X, all_data.Y, test_size=test_data_size, random_state=42
    )

    normalizer = DataNormalizer(X_train)
    X_train_norm = normalizer.normalize(X_train)
    # テストデータも同じ統計量で正規化しておく
    X_test_norm = normalizer.normalize(X_test)

    train_loader = DataLoader(X_train_norm, Y_train, batch_size=BATCH_SIZE)

    setter = ModelSetter(X_train_norm, Y_train)
    setter.setting_Flame(HIDDEN_LAYER)
    setter.setting_Function(ACT_FUNCTION, OUTPUT_FUNCTION)
    setter.setting_Coefficient(ETA, L2_LAMBDA, ALPHA)
    model = setter.create_model()
    
    # プロッターには訓練データの一部（可視化用）を渡す
    plotter = Plotter(0.1, normalizer.normalize, X_train[:500], Y_train[:500], IS_DETAIL_MODE)

    print(f"Start training: {len(X_train)} samples, {len(train_loader)} batches per epoch")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        for X_batch, Y_batch in train_loader:
            model.shift(X_batch, Y_batch)
        
        # 損失の記録と表示（毎エポックではなく一定間隔に）
        if epoch % 10 == 0:
            # 訓練データの一部で損失を近似（高速化のため）
            train_loss = model.loss(X_train_norm[:test_data_size], Y_train[:test_data_size])
            test_loss = model.loss(X_test_norm, Y_test)
            
            model.train_loss_history.append(train_loss)
            model.test_loss_history.append(test_loss)
            
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            if epoch % 50 == 0:
                plotter.show(model)

    end_time = time.time()
            
    plotter.show(model)

    # 訓練データの正解率
    Y_train_pred = np.argmax(model.predict(X_train_norm), axis=1)
    Y_train_labels = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
    train_accuracy = np.mean(Y_train_pred == Y_train_labels)
    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")

    # 最終的な正解率の計算
    Y_test_pred = np.argmax(model.predict(X_test_norm), axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test
    test_accuracy = np.mean(Y_test_pred == Y_test_labels)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    plotter.finish()

    print("time:", end_time - start_time)

if __name__ == "__main__":
    main()
