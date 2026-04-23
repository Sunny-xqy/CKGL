import torch
import yaml
from Relation_Update import DynamicGraphUpdater  # 替换为你的模型实际路径
from train_RU import DynamicTrainer  # 替换为实际的Trainer定义文件路径
import pickle
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm

with open('shareholding.yaml', 'r') as f:
    args = yaml.safe_load(f)
    args = SimpleNamespace(**args)


def load_dynamic_edges(path):
    dynamic_edge_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src, dst, rel, timestamp = map(int, line.strip().split(','))
            if timestamp not in dynamic_edge_dict:
                dynamic_edge_dict[timestamp] = []
            dynamic_edge_dict[timestamp].append((src, dst, rel, timestamp))
    return dynamic_edge_dict


def load_preserved_embeddings(path):
    att = np.load(path, allow_pickle=True)
    relation_dict = att.item()  # {'0': (num_nodes, hidden_dim), ...}

    num_relations = 0
    num_nodes = 0
    


    for rel_str, emb in tqdm(relation_dict.items()):
        rel = int(rel_str)
        num_relations = max(num_relations, rel)
        num_nodes = max(num_nodes, max(int(node) for node in emb.keys()))
        hidden_dim = len(emb[str(num_nodes)])

    print(f" 。。。num_relations。。。: {num_relations}")
    print(f" 。。。num_nodes。。。: {num_nodes}")
    print(f" 。。。hidden_dim。。。: {hidden_dim}")

    embedding_tensor = torch.zeros((num_relations+1, num_nodes+1, hidden_dim), device=args.device)

    for rel_str, emb in tqdm(relation_dict.items()):
        rel = int(rel_str)
        for node, embedding in emb.items():
            emb_tensor = torch.tensor(embedding, device=args.device)
            embedding_tensor[rel, int(node)] = emb_tensor

    return embedding_tensor, num_relations+1, num_nodes+1, hidden_dim

def main():
    device = torch.device(args.device)

    dynamic_edge_dict = load_dynamic_edges(args.dynamic_edge_path)

    preserved_embedding_path = args.preserved_node_embedding_path
    preserved_embeddings, num_relations, num_nodes, hidden_dim = load_preserved_embeddings(preserved_embedding_path)

    model = DynamicGraphUpdater(args,
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_dim=hidden_dim,
        pretrained_node_embeddings=preserved_embeddings
    )

    trainer = DynamicTrainer(args,
        model=model,
        dynamic_edge_dict=dynamic_edge_dict,
        pretrained_node_embeddings=preserved_embeddings
    )

    trainer.train()

if __name__ == "__main__":
    main()
