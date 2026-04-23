import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
import math
from tqdm import tqdm

embeddingpath = '/home/xuqingying/my_work/EGES/EKGL/embeddings/2019share_evolvogcn_train_nodeembs.csv.gz'
groundtruthpath = '/home/xuqingying/my_work/EGES/EKGL/data/groundtruth.csv'
modelpath = '/home/xuqingying/my_work/EGES/EKGL/model/evolvegcn_model.h5'

def get_data_from_csv(embedding_path, groundtruth_path, train_ratio=0.7, val_ratio=0.1, neg_ratio=3.0):
    """
    读取embedding和groundtruth，构造训练/验证/测试数据。
    增加负采样和归一化处理。
    """
    # 1. Load node embeddings
    if embedding_path.endswith(".gz"):
        embeddings_df = pd.read_csv(embedding_path, compression='gzip', index_col=0)
    else:
        embeddings_df = pd.read_csv(embedding_path, index_col=0)

    embeddings = embeddings_df.to_dict(orient='index')

    # 2. Load ground truth
    gt_df = pd.read_csv(groundtruth_path)

    # 3. 对control值进行归一化
    scaler = MinMaxScaler()
    gt_df['control'] = scaler.fit_transform(gt_df[['control']])

    # 4. 生成负样本
    all_nodes = set(embeddings.keys())
    neg_samples = []

    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Generating negative samples"):
        n1, n2 = int(row['from_id']), int(row['to_id'])
        for _ in range(int(neg_ratio)):
            neg_node = np.random.choice(list(all_nodes - {n1}))
            neg_samples.append({
                'from_id': neg_node,
                'to_id': n2,
                'control': 0.0,
                'time_code': row['time_code']
            })

    all_samples = pd.concat([gt_df, pd.DataFrame(neg_samples)], ignore_index=True)

    # 5. 准备数据
    data1, data2, controlpower, years = [], [], [], []

    for _, row in tqdm(all_samples.iterrows(), total=len(all_samples), desc="Preparing data"):
        n1, n2 = int(row['from_id']), int(row['to_id'])
        if n1 in embeddings and n2 in embeddings:
            data1.append(np.array(list(embeddings[n1].values()), dtype=np.float32))
            data2.append(np.array(list(embeddings[n2].values()), dtype=np.float32))
            controlpower.append(float(row['control']))
            years.append(row['time_code'])

    data1 = np.array(data1)
    data2 = np.array(data2)
    controlpower = np.array(controlpower).reshape(-1, 1)
    years = np.array(years)

    # 6. 按年份划分数据集
    unique_years = np.unique(years)
    train_years = unique_years[:int(len(unique_years) * train_ratio)]
    val_years = unique_years[int(len(unique_years) * train_ratio):int(len(unique_years) * (train_ratio + val_ratio))]
    test_years = unique_years[int(len(unique_years) * (train_ratio + val_ratio)):]

    train_mask = np.isin(years, train_years)
    val_mask = np.isin(years, val_years)
    test_mask = np.isin(years, test_years)

    train = [[data1[train_mask], data2[train_mask]], controlpower[train_mask]]
    valid = [[data1[val_mask], data2[val_mask]], controlpower[val_mask]]
    test = [[data1[test_mask], data2[test_mask]], controlpower[test_mask]]

    # 7. 构造按年份划分的测试集
    yearly_test = {}
    for y in unique_years:
        year_mask = (years == y)
        yearly_test[y] = [[data1[year_mask], data2[year_mask]], controlpower[year_mask]]

    return train, valid, test,  yearly_test

def evaluate_model(model, test_data, threshold=0.6):
    """
    评估模型性能，计算多个指标
    """
    # 1. 预测
    # print(test_data[0])
    test_pred = model.predict(test_data[0])
    
    # 2. 计算回归指标
    mse = mean_squared_error(test_data[1], test_pred)
    rmse = math.sqrt(mse)
    
    # 3. 计算分类指标（基于阈值）
    y_true_binary = (test_data[1] >= threshold).astype(int)
    y_pred_binary = (test_pred >= threshold).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

def evaluate_by_year(model, yearly_test, threshold=0.5):
    """
    按年份评估模型性能

    参数:
    - model: 训练好的模型，必须有 predict 方法
    - yearly_test: 按年份划分的测试集字典，来自 get_data_from_csv 的返回
    - threshold: 二分类决策阈值

    返回:
    - results: 每个年份的评估结果字典
    """
    results = {}

    for year, test in yearly_test.items():
        results[year] = evaluate_model(model, test, threshold)

    return results

def build_CoNN_model(embedding_dim):
    """
    构建 CoNN 模型：输入两个embedding向量，输出一个control分数（回归）。
    """
    input1 = Input(shape=(embedding_dim,))
    input2 = Input(shape=(embedding_dim,))
    merged = Concatenate()([input1, input2])

    x = Dense(128, activation='relu')(merged)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)  # MSE 回归任务

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error')
    return model


def train_model(train_data, val_data, embedding_dim, batch_size=512, filepath='model/CoNN_best.h5'):
    """
    训练CoNN模型。
    """
    model = build_CoNN_model(embedding_dim)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        x=train_data[0],
        y=train_data[1],
        validation_data=(val_data[0], val_data[1]),
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


if __name__ == '__main__':
    # ======== 配置路径 ========
    model_save_path = 'model/CoNN_best.h5'

    # ======== 获取数据 ========
    train, valid, test, test_years = get_data_from_csv(
        embeddingpath, 
        groundtruthpath)

    # 自动获取embedding维度
    embedding_dim = train[0][0].shape[1]

    # ======== 模型训练 ========
    model, history = train_model(train, valid, embedding_dim, batch_size=512, filepath=model_save_path)

    # ======== 测试评估 ========
    # 1. 整体评估
    overall_metrics = evaluate_model(model, test)
    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 2. 按年份评估
    year_metrics = evaluate_by_year(model, test_years)
    print("\nMetrics by Year:")
    for year, metrics in year_metrics.items():
        print(f"\nYear {year}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
