import torch
import torch.nn as nn

from utils.config import load_config


class DGCFNetwork(nn.Module):
    def __init__(self, num_users, num_items):
        cfg = load_config()
        model_cfg = cfg["model"]

        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = model_cfg["embedding_dim"]
        self.num_intents = model_cfg["num_intents"]
        self.num_layers = model_cfg["num_layers"]

        # Embedding shapes: [num_users/items, K, d]
        self.user_embeddings = nn.Parameter(
            torch.randn(num_users, self.num_intents, self.embedding_dim)
        )
        self.item_embeddings = nn.Parameter(
            torch.randn(num_items, self.num_intents, self.embedding_dim)
        )

    def forward(self):
        """
        Placeholder forward function — for now just returns user/item embeddings.
        """
        return self.user_embeddings, self.item_embeddings


def bpr_loss(user_e: torch.Tensor, pos_e: torch.Tensor, neg_e: torch.Tensor, reg_weight: float = 1e-4) -> torch.Tensor:
    """Calculate the Bayesian Personalised Ranking loss using the embeddings

    Args:
        user_e (torch.Tensor): The user embedding returned by the model.
        pos_e (torch.Tensor): The positive item (user like) embedding returned by the model.
        neg_e (torch.Tensor): The negative item (user ignored) embedding returned by the model.
        reg_weight (float, optional): The regularisation weight. Defaults to 1e-4.

    Returns:
        torch.Tensor: Returns the regularised BPR Loss
    """
    # user_e, pos_e, neg_e: [B, D] or [B, K, D]
    diff = (pos_e - neg_e) * user_e
    # Combine latent-intent and embedding dimensions into one score per sample
    # diff shape: [batch_size, num_intents, embedding_dim]
    # sum over intents and embedding dims -> [batch_size]
    prediction = diff.sum(dim=(-2, -1))
    loss = -torch.log(torch.sigmoid(prediction) + 1e-8).mean()

    # Optional L2 regularization
    reg = (user_e.norm(2).pow(2) +
           pos_e.norm(2).pow(2) +
           neg_e.norm(2).pow(2)) / user_e.size(0)

    return loss + reg_weight * reg
