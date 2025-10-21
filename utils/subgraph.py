import torch
from torch_geometric.loader import NeighborLoader

class SubgraphExtractor():
    def __init__(self):
        pass

    def get_batch_nodes(self, triplet_batch):
        
        paper_ids = triplet_batch[0]
        pos_ids   = triplet_batch[1]
        neg_ids   = triplet_batch[2]

        all_papers = list(set(paper_ids.tolist()))
        all_datasets = list(set(torch.cat((pos_ids, neg_ids), dim=0).tolist()))

        return all_papers, all_datasets

    def get_subgraph(self, paper_ids, full_graph):
        
        input_paper_nodes = torch.tensor(paper_ids)
        if input_paper_nodes.dim() == 0:
            input_paper_nodes = input_paper_nodes.unsqueeze(0)

        loader = NeighborLoader(
            data=full_graph,
            input_nodes=['paper', input_paper_nodes],
            num_neighbors=[20, 15], 
            shuffle=False,
            batch_size=len(paper_ids)
        )
        
        return next(iter(loader))  
        
    def map_global_to_local(self, triplets, batch):
        paper_nid = batch['paper'].n_id.tolist()
        dataset_nid = batch['dataset'].n_id.tolist()

        g2l_paper = {g: i for i, g in enumerate(paper_nid)}
        g2l_dataset = {g: i for i, g in enumerate(dataset_nid)}
        
        paper_idx = []
        pos_idx = []
        neg_idx = []
        for anchor_id, pos_id, neg_id in zip(*triplets):
            paper_idx.append(g2l_paper[int(anchor_id)])
            pos_idx.append(g2l_dataset[int(pos_id)])
            neg_idx.append(g2l_dataset[int(neg_id)])
            
        return paper_idx, pos_idx, neg_idx
    
    def run(self, triplet_batch, train_data):
        paper_ids, dataset_ids = self.get_batch_nodes(triplet_batch)
        batch = self.get_subgraph(paper_ids, train_data)
        paper_idx, pos_idx, neg_idx = self.map_global_to_local(triplet_batch, batch)
        
        return batch, paper_idx, pos_idx, neg_idx