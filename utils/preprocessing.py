import pandas as pd
import numpy as np
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from tqdm import tqdm
import os

class Preprocessor():
    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.device = device
        self.lm = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
        
        
    def encode(self, paper_text):
        tok_text = self.tokenizer(paper_text,
                             truncation = True,
                             max_length = 512,
                             padding = 'max_length',
                             return_tensors='pt').to(self.device)
        
        tok_text = {k: v.to(self.device) for k, v in tok_text.items()}
    
        with torch.no_grad():
            outputs = self.lm(**tok_text)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return embeddings
    
    def combine_text(self, dataset_content, unseen_train_data, test_data):
        
        dataset_texts = dataset_content[
            ['dataset_title', 'dataset_name', 'dataset_content', 'dataset_structured_data']
        ].agg('[SEP]'.join, axis=1)
        paper_texts = unseen_train_data[
            ['keyphrase_query', 'query', 'title', 'abstract']
        ].agg('[SEP]'.join, axis=1)
        test_texts = test_data[
            ['query', 'keyphrase_query', 'abstract']
        ].agg('[SEP]'.join, axis=1)
        
        return dataset_texts, paper_texts, test_texts
        
    def text_to_embedding(self, texts):
        tqdm.pandas()
        embeddings = texts.progress_apply(lambda x: self.encode(x))
        return embeddings
    
    def text_to_tensor(self, dataset_texts, paper_texts, test_texts):
        dataset_embeddings = self.text_to_embedding(dataset_texts)
        paper_embeddings = self.text_to_embedding(paper_texts)
        test_embeddings = self.text_to_embedding(test_texts)
        
        dataset_embeddings_tensor = torch.vstack(list(dataset_embeddings))
        paper_embeddings_tensor = torch.vstack(list(paper_embeddings))
        test_embeddings_tensor = torch.vstack(list(test_embeddings))
        
        all_paper_embeddings_tensor = torch.cat((paper_embeddings_tensor, test_embeddings_tensor), dim=0)
        
        return dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor
        


    def load_data(self, original_training_path, unseen_training_path, unseen_testing_path, dataset_information_path):
        df_ori = pd.read_json(original_training_path, lines=True)
        df_unseen = pd.read_json(unseen_training_path, lines=True)
        df_test = pd.read_json(unseen_testing_path, lines=True)
        df_dataset_information = pd.read_json(dataset_information_path, lines=True)
        
        df_ori.drop_duplicates(subset=['paper_id'], inplace=True)
        df_unseen.drop_duplicates(subset=['paper_id'], inplace=True)

        df_test = df_test.map(lambda x: np.nan if x == [] else x)
        df_test = df_test.dropna(ignore_index = True)
        df_test[['query', 'keyphrase_query', 'abstract']] = df_test[['query', 'keyphrase_query', 'abstract']].map(lambda x: x.lower())
        
        pprIdx = {paper_id: idx for idx, paper_id in enumerate(df_ori['paper_id'])}
        pprIdx_test = {index: idx for idx, (index, row) in enumerate(df_test.iterrows())}
        
        df_unseen['title'] = df_unseen['title'].apply(lambda x: x.lower())

        return df_ori, df_unseen, df_test, pprIdx, pprIdx_test, df_dataset_information
    
    def dataset_to_label(self, dataset_positive, dataset_negative, test_dataset, unseen_train_data, test_data):
        label = {} #for all dataset in the train data
        label_counter = 0

        for data_list in dataset_positive + dataset_negative:
            for data_name in data_list:
                if data_name not in label:
                    label[data_name] = label_counter
                    label_counter += 1
                    
        unseen_train_data['positive label'] = [[label[ds] for ds in datasets] for datasets in unseen_train_data['positives']]
        unseen_train_data['negative label'] = [[label[ds] for ds in datasets] for datasets in unseen_train_data['negatives']]
        test_data['dataset label'] = [[label[ds] for ds in datasets] for datasets in test_dataset]
        
        return label
    
    def coo_construction_train(self, unseen_train_data, index_train):
        edges_p2d = []
        all_triplets = []

        for _, row in unseen_train_data.iterrows():
            source_idx = index_train[row['paper_id']]
            positives = row['positive label']
            negatives = row['negative label']
            

            pos_edges = [[source_idx, pid] for pid in positives]
            neg_edges = [[source_idx, nid] for nid in negatives]

            edges_p2d.extend(pos_edges + neg_edges)
    

            all_triplets.extend([[source_idx, pos, neg] for pos in positives for neg in negatives])
            
        positive_negative_index = torch.tensor(edges_p2d, dtype=torch.long).t().contiguous()
            
        return positive_negative_index, all_triplets
    
    def coo_construction_test(self, test_data, index_test):
        edges_p2d_test = []

        for _, row in test_data.iterrows():
            source_node_index = index_test[_]+ 17397
            dataset =  row['dataset label']
            
            edges = [[source_node_index, pid] for pid in dataset]
            
            edges_p2d_test.extend(edges)
                
        edge_index_p2d_test = torch.tensor(edges_p2d_test, dtype=torch.long).t().contiguous()

        return edge_index_p2d_test
    
    def cleaning_text(self):
        
        pattern = (
            r'\\\(.*?\\\)|'  # LaTeX inline math (\(...\))
            r'\r\n|'  # Line breaks
            r'\*\*|'  # Asterisks
            r'\$.*?\$|'  # LaTeX inline math ($...$)
            r'\\\[.*?\\\]|'  # LaTeX display math (\[...\])
            r'https?://\S+|'  # URLs starting with http:// or https://
            r'www\.\S+|'  # URLs starting with www.
            r'ftp://\S+|'  # URLs starting with ftp://
            r'\\begin\{equation\}.*?\\end\{equation\}|'  # LaTeX display math (\begin{equation}...\end{equation})
            r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?|' # LaTeX commands
            r'\\langle.*?\\rangle|'  # LaTeX angle brackets (\langle...\rangle)
            r'https?://[^\s]+(?:[\s\.,]|$)|'  # Match http or https URLs, followed by space, dot, or end of string
            r'www\.[^\s]+(?:[\s\.,]|$)'  # Match URLs starting with www., followed by space, dot, or end of string
            r'\[Image Source\: \[.*?\]|'
            r'\(Image Source\: \[|'
            r'\[Image Source\: \[|'
            r'\(Source\: \[.*?\]|'
            r'Source\: \[|'  
            r'\(\s*/paper/[^)]+\s*\)'
            
        )  
        
        return (lambda x: re.sub(pattern, '', x))
    
    def cleaning_dataset(self, text):
        pattern_dataset = (
            r'\\\(.*?\\\)|'  # LaTeX inline math (\(...\))
            r'\r\n|'  # Line breaks
            r'\*\*|'  # Asterisks
            r'\*|'  # Asterisks
            r'\$.*?\$|'  # LaTeX inline math ($...$)
            r'\\\[.*?\\\]|'  # LaTeX display math (\[...\])
            r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?|' # LaTeX commands
            r'\n|'
            r'\n+|'

            r'\(https?://\S+\)|'  # URLs starting with http:// or https://
            r'(Source:|Image Source:|Image:|NOTE: ).*|'
            r'\[\'|'
            r'\'\]|'
            r'\{|'
            r'\}|'
            
        )  
        
        # Remove unwanted patterns
        cleaned = re.sub(pattern_dataset, '', text)
        # Condense multiple whitespace characters into a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove leading and trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def dataset_content_extraction(self, dataset_information, label):
        
        dataset_idcontent_all = []
        
        for _, row in dataset_information.iterrows():
            dataset_name = row['id']
            dataset_content = row['contents']
            dataset_title = row['title']
            dataset_structured_data = row['structured_info']
            if dataset_name in label.keys():
                dataset_target_id = label[dataset_name]
                dataset_idcontent_all.append((dataset_target_id, dataset_name, dataset_content, dataset_title, dataset_structured_data))

        dataset_content = pd.DataFrame(dataset_idcontent_all, columns=['dataset_target_id', 'dataset_name', 'dataset_content', 'dataset_title', 'dataset_structured_data' ])

        dataset_content.sort_values(by='dataset_target_id', inplace=True, ignore_index=True)
        dataset_content.drop_duplicates(subset=['dataset_target_id'], inplace=True, ignore_index=True)
        dataset_content[['dataset_title', 'dataset_name']] = dataset_content[['dataset_title', 'dataset_name']].map(lambda x: x.lower())
        
        return dataset_content         

        
        
    
    def run(self, original_training_path, unseen_training_path, unseen_testing_path, dataset_information_path):
        ori_train_data, unseen_train_data, test_data, index_train, index_test, dataset_information = self.load_data(
            original_training_path, 
            unseen_training_path, 
            unseen_testing_path, 
            dataset_information_path
        )
        
        label = self.dataset_to_label(
            ori_train_data['positives'], 
            ori_train_data['negatives'],
            test_data['documents'],
            unseen_train_data,
            test_data
            )
        
        positive_negative_index, all_triplets = self.coo_construction_train(
            unseen_train_data, 
            index_train
            )   
        
        
        
        edge_index_p2d_test = self.coo_construction_test(
            test_data, 
            index_test
            )      
        
        
        print("COO construction done....")

        unseen_train_data['abstract'] = unseen_train_data['abstract'].apply(self.cleaning_text()).apply(lambda x: x.lower())
        test_data['abstract'] = test_data['abstract'].apply(self.cleaning_text()).apply(lambda x: x.lower())
        
        unseen_train_data['query'] = unseen_train_data.apply(lambda row: row['query'].replace('[DATASET]', str(row['positives'])), axis=1)
        
        dataset_content = self.dataset_content_extraction(
            dataset_information, 
            label
            )
        
        dataset_content[['dataset_content', 'dataset_structured_data']] = dataset_content[['dataset_content', 'dataset_structured_data']].map(self.cleaning_dataset).apply(lambda x: x.str.lower())
        unseen_train_data['query'] = unseen_train_data['query'].apply(self.cleaning_dataset).apply(lambda x: x.lower())
        
        dataset_texts, paper_texts, test_texts = self.combine_text(
            dataset_content, 
            unseen_train_data, 
            test_data
            )
        
        print("Converting Text to Embeddings...")
        
        if os.path.exists('dataset_embeddings_tensor.pt') and os.path.exists('all_paper_embeddings_tensor.pt') and os.path.exists('paper_embeddings_tensor.pt'):
            dataset_embeddings_tensor = torch.load('dataset_embeddings_tensor.pt')
            all_paper_embeddings_tensor = torch.load('all_paper_embeddings_tensor.pt')
            paper_embeddings_tensor = torch.load('paper_embeddings_tensor.pt')
        else:    
            dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor = self.text_to_tensor(
                dataset_texts, 
                paper_texts, 
                test_texts
                )
        
            torch.save(dataset_embeddings_tensor, 'dataset_embeddings_tensor.pt')
            torch.save(all_paper_embeddings_tensor, 'all_paper_embeddings_tensor.pt')
            torch.save(paper_embeddings_tensor, 'paper_embeddings_tensor.pt')

        return dataset_embeddings_tensor, all_paper_embeddings_tensor, paper_embeddings_tensor, positive_negative_index, edge_index_p2d_test, all_triplets