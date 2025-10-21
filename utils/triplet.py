import torch
from torch_geometric.loader import DataLoader
  
    
class TripletLoader():
    def __init__(self):
        pass
    
    def create_triplet_loader(self, all_triplets, g):
        triplet_loader = DataLoader(
            TripletDataset(all_triplets),
            batch_size=8192,
            shuffle=True,
            generator=g,
            )
        return triplet_loader
        
    


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]