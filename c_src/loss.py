import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # デバッグプリントを追加
        print(f"sim shape: {sim.shape}")
        print(f"batch_size: {self.batch_size}")
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # デバッグプリントを追加
        print(f"sim_i_j shape: {sim_i_j.shape}")
        print(f"sim_j_i shape: {sim_j_i.shape}")
        
        positives = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        mask = torch.eye(N, dtype=bool).to(sim.device)
        negatives = sim[~mask].view(N, -1)
        labels = torch.zeros(N).to(positives.device).long()
        logits = torch.cat((positives, negatives), dim=1)
        loss = self.criterion(logits, labels)
        return loss / N