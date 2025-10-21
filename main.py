
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


from utils.preprocessing import Preprocessor
from utils.evaluation import ranking_evaluation
from utils.triplet import TripletLoader
from utils.loss import text_cl, margin_ranking, l2_reg_loss
from model import *
from utils.subgraph import SubgraphExtractor

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch
import numpy as np
import random
import faiss
from collections import defaultdict
import argparse

from tqdm import tqdm



original_training_path =  './dataset/train_data.jsonl'
unseen_training_path = './dataset/unseen_dataset_train.jsonl'
unseen_testing_path = './dataset/unseen_dataset_test.jsonl'
dataset_information_path = './dataset/dataset_search_collection.jsonl'

encoder_map = {
    'sage': SAGE,
    'rgcn': RGCN,
    'gat': GAT,
}
 
class Trainer():
    def __init__(self, encoder_name='gat', device='cuda', learning_rate=5e-3):
        self.seed_num = 1
        self.device = device
        self.tripletloader = TripletLoader()
        self.subgraph = SubgraphExtractor()
        self.learning_rate = learning_rate
        self.encoder_name = encoder_name.lower()
        self.hidden_channels = 256
        
        self.encoder_class = encoder_map.get(self.encoder_name)
        if self.encoder_class is None:
            raise ValueError(f"Unknown encoder: {self.encoder_name}")       
        self.warmup = 5
        self.base_text_cl_rate = 0.8
        self.margin_loss_rate = 0.8
        self.reg = 1e-4
        self.epochs = 40
        self.bestPerformance = []
    
    def set_seed(self, seed):   
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
    def preprocess(self):
        print("Preprocessing Data...")
        preprocessor = Preprocessor(self.device)
        dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index, edge_index_p2d_test, all_triplets = preprocessor.run(
            original_training_path, 
            unseen_training_path, 
            unseen_testing_path,
            dataset_information_path,
        )
        
        return dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index, edge_index_p2d_test, all_triplets
    
    def graph_construction(self, dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index):
        train_data = HeteroData()
        train_data['paper'].x = paper_embeddings_tensor
        train_data['dataset'].x = dataset_embeddings_tensor
        train_data['paper', 'use', 'dataset'].edge_index = positive_negative_index
        train_data['paper', 'use', 'dataset'].edge_label_index = train_data['paper', 'use', 'dataset'].edge_index

        train_data = T.ToUndirected()(train_data)
        
        test_data = HeteroData()
        test_data['paper'].x = all_paper_embeddings_tensor
        test_data['dataset'].x = dataset_embeddings_tensor
        test_data['paper', 'use', 'dataset'].edge_index = positive_negative_index
        test_data['paper', 'use', 'dataset'].edge_label_index = test_data['paper', 'use', 'dataset'].edge_index

        test_data = T.ToUndirected()(test_data)
        
        return train_data.to(self.device), test_data.to(self.device)
    
    
    def setup(self, warmup_epochs=5):
        self.model = Model(encoder_name=self.encoder_name, hidden_channels=self.hidden_channels, encoder=self.encoder_class, data=self.train_data).to(self.device) 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        linear_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max= self.epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        
    def run(self): 
        self.set_seed(self.seed_num)
        torch.use_deterministic_algorithms(True)
        self.g = torch.Generator()
        self.g.manual_seed(self.seed_num)       
        dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index, edge_index_test, all_triplets = self.preprocess()
        print("Preprocessing Completed.")
        self.train_data, self.test_data = self.graph_construction(dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index)
        self.triplet_loader = self.tripletloader.create_triplet_loader(all_triplets, self.g)
        
        self.setup()
        ground_truth = self.ground_truth_dict(edge_index_test)
        print("Starting Training...")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train()
            print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.4f}")
            result = self.test(ground_truth, edge_index_test)
            
            if self.bestPerformance:
                count = sum(1 if self.bestPerformance[1][k] > result[k] else -1 for k in result)
                if count < 0:
                    early_stopping = 0
                    self.bestPerformance = [epoch, result]
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'performance': result,
                    }, 'best_model.pt')
                
                    print(f"Saved best model at epoch {epoch}")
                    
                else:
                    early_stopping += 1

            else:
                self.bestPerformance = [epoch, result]
                early_stopping = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'performance': result,
                    }, 'best_model.pt')
                
                print(f"Saved initial run model at epoch {epoch}")
                
            print('-' * 80)
            print(f'Real-Time Ranking Performance (Top-5 Dataset Recommendation)')
            measure_str = ', '.join([f'{k}: {v}' for k, v in result.items()])
            print(f'*Current Performance*\nEpoch: {epoch}, {measure_str}')
            bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
            print(f'*Best Performance*\nEpoch: {self.bestPerformance[0]}, {bp}')
            print('-' * 80)
            
            if early_stopping >= 7:
                    print('EARLY STOPPING TRIGGERED!!!!!')
                    break
            
            
        
    def train(self):
        self.model.train()
        total_loss = 0.0
        total_examples = 0   
        
        text_cl_rate = min(self.epochs / self.warmup, 1.0) * self.base_text_cl_rate
        
        
        for n, batch_triplets in (pbar := tqdm(enumerate(self.triplet_loader))):
            
            
            batch, paper_idx, pos_idx, neg_idx = self.subgraph.run(batch_triplets, self.train_data)


            
            rec_paper_emb, rec_dataset_emb = self.model(
                batch.x_dict, 
                batch.edge_index_dict, 
            )

            paper_emb, pos_emb, neg_emb = rec_paper_emb[paper_idx], rec_dataset_emb[pos_idx], rec_dataset_emb[neg_idx]
            
            text_cl_loss = text_cl_rate * text_cl(paper_emb, pos_emb)

            
            margin_ranking_loss = self.margin_loss_rate * margin_ranking(paper_emb, pos_emb, neg_emb)

            batch_loss =  l2_reg_loss(self.reg, paper_emb, pos_emb) + text_cl_loss + margin_ranking_loss
               
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += batch_loss.item()
            total_examples += 1
            
        return total_loss / total_examples
    
    def ground_truth_dict(self, edge_index_test):
        paper_to_dataset = defaultdict(set)
        for p, i in zip(edge_index_test[0].tolist(), edge_index_test[1].tolist()):
            paper_to_dataset[p].add(int(i))
            
        ground_truth = []

        for paper, dataset in paper_to_dataset.items():
            ground_truth.append(dataset)
            
        ground_truth = [list(items) for items in ground_truth]
        
        return ground_truth
        
    
    def test(self, ground_truth, edge_index_test, k=5):
        results = []

        self.model.eval()
        with torch.no_grad():
            ori_paper_emb, dataset_emb = self.model(
                self.test_data.x_dict, 
                self.test_data.edge_index_dict,
            )
            
            test_papers_idx = edge_index_test[0]

            paper_nids, inv = torch.unique(test_papers_idx, sorted=True, return_inverse=True)

            paper_emb = ori_paper_emb[paper_nids] 

            # --- Convert to numpy and FAISS-compatible format ---
            paper_np = paper_emb.cpu().numpy().astype('float32')
            dataset_np = dataset_emb.cpu().numpy().astype('float32')

            # --- Build FAISS index ---
            dim = dataset_np.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product search
            index.add(dataset_np)

            # --- FAISS Search ---
            scores, indices = index.search(paper_np, k)  # shape: (num_queries, k)
            scores = torch.tensor(scores)
            results = torch.tensor(indices)
            print(results[:10])

            # Convert to the format: {index: {item: 1, ...}}
            origin_dict = {i: {item: 1 for item in items} for i, items in enumerate(ground_truth)}

            # Convert res_tensor to: {index: [(item, 0.0), ...]}
            res_dict = {i: [(int(item), 0.0) for item in row.tolist()] for i, row in enumerate(results)}

            measure = ranking_evaluation(origin_dict, res_dict, [5])

            performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

            return performance
        
def print_models(models):
    all_models = sum(models.values(), [])
    print("Available models:")
    for model in all_models:
        print(f"- {model}")
    return all_models
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dataset recommendation model.")

    parser.add_argument('--encoder', type=str, default='gat', choices=encoder_map.keys(),
                        help='Encoder architecture to use (default: gat)')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate (default: 5e-3)')

    args = parser.parse_args()

    trainer = Trainer(encoder_name=args.encoder, learning_rate=args.lr)
    trainer.run()
    
           