import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import random
embeddingpath = '/home/xuqingying/my_work/EGES/EKGL/embeddings/2021share_evolvogcn_train_nodeembs.csv.gz'
groundtruthpath = '/home/xuqingying/my_work/EGES/EKGL/data/groundtruth.csv'

embeddingpath_pt = '/home/xuqingying/my_work/EGES/EKGL/baseline/Inductive-representation-learning-on-temporal-graphs'

def load_node_embs_from_csv(emb_path):
    df = pd.read_csv(emb_path, header=None, compression='gzip')
    embs = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float)
    return embs

def load_node_embs_from_pt(emb_path,year):
    emb_path1 = f'{emb_path}/embeddings_at_time_{int(year)}.pt'
    embs = torch.load(emb_path1)
    return embs

def compute_link_pred_f1_by_year(embs,  edge_csv, year=None, threshold=0.3, negative_samples_ratio=3.0):
    df = pd.read_csv(edge_csv)

    # 对 'control' 列进行归一化（min-max）
    # if 'control' in df.columns:
    #     df['control'] = (df['control'] - df['control'].min()) / (df['control'].max() - df['control'].min())

    # 如果指定年份，则筛选该年份数据
    if year is not None:
        df = df[df['time_code'] == year]

    y_true = []
    y_pred = []
    nodes = list(range(len(embs)))
    positive_pairs = set()

    for _, row in df.iterrows():
        from_id = int(row['from_id'])
        to_id = int(row['to_id'])
        positive_pairs.add((from_id, to_id))
        positive_pairs.add((to_id, from_id))  # 无向图

    # 负采样
    negative_pairs = set()
    while len(negative_pairs) < len(positive_pairs) * negative_samples_ratio:
        from_id = random.choice(nodes)
        to_id = random.choice(nodes)
        if from_id != to_id and (from_id, to_id) not in positive_pairs and (to_id, from_id) not in positive_pairs:
            negative_pairs.add((from_id, to_id))

    # 合并样本
    all_pairs = list(positive_pairs) + list(negative_pairs)

    for from_id, to_id in all_pairs:
        sim = F.cosine_similarity(embs[from_id].unsqueeze(0), embs[to_id].unsqueeze(0)).item()
        true = 1 if (from_id, to_id) in positive_pairs else 0
        pred = 1 if sim > threshold else 0
        y_true.append(true)
        y_pred.append(pred)

    return f1_score(y_true, y_pred)

if __name__ == "__main__":
    # embs = load_node_embs_from_csv(embeddingpath)
    f1_list = []
    for year in range(0,7):
        embs_pt = load_node_embs_from_pt(embeddingpath_pt,year)
        f1 = compute_link_pred_f1_by_year(embs_pt,groundtruthpath, year=year, threshold=1)
        f1_list.append(f1)
        print(f'year {year} f1: {f1}')
    print(f1_list)