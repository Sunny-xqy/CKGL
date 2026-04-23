import torch
import torch.nn as nn

class DynamicGraphUpdater(nn.Module):
    def __init__(self, args, num_nodes, num_relations, hidden_dim, pretrained_node_embeddings=None):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.time_buckets = args.time_buckets
        self.device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)

        if pretrained_node_embeddings is not None:
            print(pretrained_node_embeddings.shape)
            assert pretrained_node_embeddings.shape == (self.num_relations, self.num_nodes, self.hidden_dim)
            self.prev_embeddings = pretrained_node_embeddings.clone().detach().to(self.device) #(num_relations, num_nodes, hidden_dim)
        else:
            self.prev_embeddings = torch.randn(self.num_relations, self.num_nodes, self.hidden_dim, device=self.device)

        self.relation_encoders = nn.ModuleList([
            nn.Linear(self.time_buckets, self.hidden_dim) for _ in range(2 * self.num_relations)
        ])
        self.relation_grus = nn.ModuleList([
            nn.GRUCell(2 * self.hidden_dim, self.hidden_dim) for _ in range(self.num_relations)
        ])

        self.M_r = nn.ParameterList([
            nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim)) for _ in range(self.num_relations)
        ])
        self.alpha_r = nn.Parameter(torch.ones(self.num_relations))

    def encode_time(self, time_diff):
        bucket = min(time_diff, self.time_buckets - 1)
        one_hot = torch.zeros(self.time_buckets, device=self.device)
        one_hot[bucket] = 1.0
        return one_hot
    
    def forward(self, edge_list, current_time):
        updated_embeddings = self.prev_embeddings.clone()

        for src, dst, rel, t_edge in edge_list:
            src, dst, rel = int(src), int(dst), int(rel)
            abs_rel = abs(rel)

            max_index = max(src, dst)
            if max_index >= self.num_nodes:
                new_num_nodes = max_index + 1
                delta = new_num_nodes - self.num_nodes

                padding = torch.randn(
                    self.num_relations, delta, self.hidden_dim, device=self.device
                )
                self.prev_embeddings = torch.cat([self.prev_embeddings, padding], dim=1)
                updated_embeddings = torch.cat([updated_embeddings, padding.clone()], dim=1)
                self.num_nodes = new_num_nodes

            time_diff = current_time - t_edge
            time_encoding = self.encode_time(time_diff).unsqueeze(0)

            rel_enc_idx = rel if rel >= 0 else self.num_relations + abs_rel
            encoded = self.relation_encoders[rel_enc_idx](time_encoding)

            u_j = self.prev_embeddings[abs_rel, dst].unsqueeze(0)
            u_i = self.prev_embeddings[abs_rel, src].unsqueeze(0)

            x_ir = torch.cat([encoded + u_j, u_i], dim=-1)
            h_prev = self.prev_embeddings[abs_rel, src].unsqueeze(0)
            h_new = self.relation_grus[abs_rel](x_ir, h_prev)

            updated_embeddings[abs_rel, src] = h_new.squeeze(0)

        return updated_embeddings,self.num_nodes

    def propagate(self, updated_embeddings):
        # updated_embeddings shape: (num_relations, num_nodes, hidden_dim)
        final_embed = torch.zeros(self.num_nodes, self.hidden_dim, device=self.device)

        for r in range(self.num_relations):
            transformed = updated_embeddings[r] @ self.M_r[r]  # shape: (num_nodes, hidden_dim)
            final_embed += self.alpha_r[r] * transformed

        return final_embed
