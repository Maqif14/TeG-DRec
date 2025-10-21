import torch
import torch.nn.functional as F

text_temp = 0.08
    
def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def text_cl(paper, dataset):
    
    loss = InfoNCE(paper, dataset, text_temp)
    
    return loss

def margin_ranking(paper_emb, pos_emb, neg_emb):
    # Compute scores
    pos_score = torch.sum(paper_emb * pos_emb, dim=1)
    neg_score = torch.sum(paper_emb * neg_emb, dim=1)
    

    # Ranking target: pos_score should be greater than neg_score
    target = torch.ones_like(pos_score)

    ranking_loss_fn = torch.nn.MarginRankingLoss(margin=1)
    ranking_loss = ranking_loss_fn(pos_score, neg_score, target)# Compute scores
    
    return ranking_loss