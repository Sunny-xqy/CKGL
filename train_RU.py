import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

class DynamicTrainer:
    def __init__(self, args, model, dynamic_edge_dict,pretrained_node_embeddings=None):
        self.model = model.to(args.device)
        self.dynamic_edge_dict = dynamic_edge_dict
        self.device = args.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()  # 二分类目标
        self.batch_size = args.batch_size
        self.pretrained_node_embeddings = pretrained_node_embeddings
        self.output_path = args.output_path
        self.epochs = args.epochs


    def create_data_loader(self, edge_list):
        # 将每个边转化为张量
        src_nodes = torch.tensor([src for src, _, _, _ in edge_list], dtype=torch.long)
        dst_nodes = torch.tensor([dst for _, dst, _, _ in edge_list], dtype=torch.long)
        relations = torch.tensor([rel for _, _, rel, _ in edge_list], dtype=torch.long)
        times = torch.tensor([t_edge for _, _, _, t_edge in edge_list], dtype=torch.float32)

        dataset = TensorDataset(src_nodes, dst_nodes, relations, times)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        num_nodes = self.model.num_nodes
        num_relations = self.model.num_relations
        hidden_dim = self.model.hidden_dim

        if self.pretrained_node_embeddings is not None:
            prev_embeddings = self.pretrained_node_embeddings.to(self.device)
        else:
            prev_embeddings = torch.zeros(num_nodes, num_relations, hidden_dim, device=self.device)

        for epoch in range(self.epochs):
            total_loss = 0.0
            # 顺序遍历每个时间步
            for current_time in sorted(self.dynamic_edge_dict.keys()):
                edge_list = self.dynamic_edge_dict[current_time]

                # 创建数据加载器，每个时间步处理一个batch
                data_loader = self.create_data_loader(edge_list)

                # 在一个batch中进行训练
                for batch in data_loader:
                    src_nodes, dst_nodes, relations, times = batch
                    src_nodes, dst_nodes, relations, times = src_nodes.to(self.device), dst_nodes.to(self.device), relations.to(self.device), times.to(self.device)

                    # 获取当前时间步的更新后的节点嵌入
                    updated_embeddings,num_nodes = self.model.forward(edge_list, current_time)

                    # 构造正样本对 (src, rel)
                    pos_pairs = set((src, abs(rel)) for src, _, rel, _ in edge_list)
                    print('Positive pairs constructed, length: ', len(pos_pairs))
                    # 采样负样本
                    all_nodes = list(range(num_nodes))
                    all_rels = list(range(num_relations))
                    neg_pairs = set()
                    while len(neg_pairs) < len(pos_pairs):
                        i = random.choice(all_nodes)
                        r = random.choice(all_rels)
                        if (i, r) not in pos_pairs:
                            neg_pairs.add((i, r))
                    print('Negative pairs sampled, length: ', len(neg_pairs))
                    # 合并正负样本
                    all_pairs = list(pos_pairs) + list(neg_pairs)
                    labels = torch.cat([torch.ones(len(pos_pairs)), torch.zeros(len(neg_pairs))]).to(self.device)
                    print('All pairs constructed, length: ', len(all_pairs))
                    # # 获取预测值
                    # preds = []
                    # for i, r in all_pairs:
                    #     emb = updated_embeddings[r, i]
                    #     score = torch.sum(emb)  # 可替换为 dot(w, emb) 或 FFN(emb)
                    #     preds.append(score)
                    # preds = torch.stack(preds)
                    
                    rows = torch.tensor([r for i, r in all_pairs], device=updated_embeddings.device)
                    cols = torch.tensor([i for i, r in all_pairs], device=updated_embeddings.device)

                    selected_embs = updated_embeddings[rows, cols]
                    preds = torch.sum(selected_embs, dim=1)  # 或其他得分方式
    
                    loss = self.criterion(preds, labels)
                    print('Loss calculated, loss: ', loss.item())
                    # 反向传播

                    # num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    # print(f'Total trainable parameters: {num_params}')
                    # print(next(self.model.parameters()).device)

                    self.optimizer.zero_grad()
                    print('Zero grad done')

                    loss.backward()
                    print('Backward done')

                    self.optimizer.step()
                    print('Step done')
                    # 更新历史嵌入
                    prev_embeddings = updated_embeddings.detach()
                    torch.save(prev_embeddings, self.output_path)
                    total_loss += loss.item()
                    print(f"[Batch Loss: {loss.item()}")

            # 每个epoch打印一次损失
            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(self.dynamic_edge_dict):.4f}")
