from collections import defaultdict


edges_relation = '/home/xuqingying/my_work/EGES/EKGL/data/edges_relation.txt'
edges_dynamic = '/home/xuqingying/my_work/EGES/EKGL/data/edges_dynamic.txt'

class EdgeDynamicsProcessor:
    def __init__(self, input_path):
        self.input_path = input_path
        self.time_to_edges = defaultdict(set)

    def parse_edge(self, line):
        """解析一行边数据为 ((src, dst, type), time)"""
        src, dst, etype, t = map(int, line.strip().split(','))
        return (src, dst, etype), t

    def load_edges(self):
        """读取所有边并按时间归类"""
        with open(self.input_path, 'r') as f:
            for line in f:
                edge, t = self.parse_edge(line)
                self.time_to_edges[t].add(edge)

    def generate_dynamic_edges(self):
        """生成动态边：记录每个时间步的新增/删除边"""
        dynamic_edges = []
        all_times = sorted(self.time_to_edges.keys())

        for i in range(len(all_times) - 1):
            t_now = all_times[i]
            t_next = all_times[i + 1]
            edges_now = self.time_to_edges[t_now]
            edges_next = self.time_to_edges[t_next]

            # 消失的边（在 t_now 存在，在 t_next 消失）
            for edge in edges_now - edges_next:
                src, dst, _ = edge
                dynamic_edges.append(f"{src},{dst},-1,{t_next}")

            # 新增的边（在 t_next 出现，在 t_now 没有）
            for edge in edges_next - edges_now:
                src, dst, etype = edge
                dynamic_edges.append(f"{src},{dst},{etype},{t_next}")

        return dynamic_edges

    def write_output(self, output_path, dynamic_edges):
        with open(output_path, 'w') as f:
            for line in dynamic_edges:
                f.write(line + '\n')

    def process(self, output_path):
        """主流程函数"""
        self.load_edges()
        dynamic_edges = self.generate_dynamic_edges()
        self.write_output(output_path, dynamic_edges)



if __name__ == "__main__":
    processor = EdgeDynamicsProcessor(edges_relation)
    processor.process(edges_dynamic)