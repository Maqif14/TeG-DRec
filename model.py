import torch
from torch_geometric.nn import GATConv, to_hetero, HeteroConv, RGCNConv, SAGEConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads=2):
        super().__init__()
        self.heads = heads
        
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        self.conv3 = GATConv((-1, -1), hidden_channels, heads=heads, concat=True, add_self_loops=False)
        
    def forward(self, x, edge_index):
        

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu() 
        x = self.conv3(x, edge_index) 

        return x
       
class RGCN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, input_dim=768):
        super().__init__()

        self.metadata = metadata  # (node_types, edge_types)

        # First layer: one RGCNConv per edge type
        self.conv1 = HeteroConv({
            rel: RGCNConv(input_dim, hidden_channels, num_relations=2)
            for rel in metadata[1]
        }, aggr='sum')

        # Second layer
        self.conv2 = HeteroConv({
            rel: RGCNConv(hidden_channels, hidden_channels, num_relations=2)
            for rel in metadata[1]
        }, aggr='sum')

        # Layer norms
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)

        # Residual projections
        self.res_proj1 = (
            torch.nn.Linear(input_dim, hidden_channels)
            if input_dim != hidden_channels else torch.nn.Identity()
        )
        self.res_proj2 = torch.nn.Identity()

    def forward(self, x_dict, edge_index_dict):
        edge_type_dict = {
            ('paper', 'use', 'dataset'): torch.full(
                (edge_index_dict[('paper', 'use', 'dataset')].size(1),),
                0,  # relation ID for "use"
                dtype=torch.long,
                device=edge_index_dict[('paper', 'use', 'dataset')].device
            ),
            ('dataset', 'rev_use', 'paper'): torch.full(
                (edge_index_dict[('dataset', 'rev_use', 'paper')].size(1),),
                1,  # relation ID for "rev_use"
                dtype=torch.long,
                device=edge_index_dict[('dataset', 'rev_use', 'paper')].device
            ),
        }

        

        out_dict = self.conv1(x_dict, edge_index_dict, edge_type_dict)
        x_dict = {
            node_type: self.norm1(F.relu(out + self.res_proj1(x_dict[node_type])))
            for node_type, out in out_dict.items()
        }

        out_dict = self.conv2(x_dict, edge_index_dict, edge_type_dict)
        x_dict = {
            node_type: self.norm2(F.relu(out + self.res_proj2(x_dict[node_type])))
            for node_type, out in out_dict.items()
        }

        return x_dict
    
    
class SAGE(torch.nn.Module):    
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu() 
        x = self.conv3(x, edge_index) 
        return x
    

class TripletDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, encoder_name, input_dim = 1280):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_name = encoder_name
        
        if self.encoder_name != 'GAT':
            self.input_dim = 1024
        
        self.paper_proj = torch.nn.Linear(self.input_dim, hidden_channels)
        self.dataset_proj = torch.nn.Linear(self.input_dim, hidden_channels)
        
        
    def forward(self, rec_x_dict, x_dict):
        paper_emb = self.paper_proj(torch.cat((rec_x_dict['paper'], x_dict['paper']), dim=1))
        dataset_emb = self.dataset_proj(torch.cat((rec_x_dict['dataset'], x_dict['dataset']), dim=1))
        return paper_emb, dataset_emb

class Model(torch.nn.Module):
    def __init__(self, encoder_name, hidden_channels, encoder, data):
        super().__init__()
        self.data = data
        self.encoder_name = encoder_name.upper()
        if self.encoder_name in ['GAT', 'SAGE']:
            self.encoder = encoder(hidden_channels)
            self.encoder = to_hetero(self.encoder, self.data.metadata(), aggr='sum')
        else:
            self.encoder = encoder(hidden_channels, metadata=self.data.metadata())
            
        self.decoder = TripletDecoder(hidden_channels, encoder_name=self.encoder_name)

    def forward(self, x_dict, edge_index_dict, cl=False): 
        
        rec_x_dict = self.encoder(x_dict, edge_index_dict)
        rec_x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in rec_x_dict.items()}
        
        x_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}
        
        return self.decoder(rec_x_dict, x_dict)
