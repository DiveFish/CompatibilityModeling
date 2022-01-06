import torch
from torch import FloatTensor


class LogisticRegression(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, value_num: int = 0, embed_dim: int = 0
    ):
        """
        A PyTorch neural network model that represents the
        functionality of a Logistic Regression classifier.
        :param input_dim: number of features
        :param output_dim: number of output classes
        """
        super(LogisticRegression, self).__init__()
        self.value_num = value_num
        self.embed_dim = embed_dim

        # For embedding without pretrained embeddings.
        if self.value_num and self.embed_dim:
            self.embed = torch.nn.Embedding(value_num, embed_dim)
            self.linear = torch.nn.Linear(input_dim * embed_dim, output_dim)
        else:
            self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, pp_head_data: FloatTensor):
        """
        Do the forward pass of the model.
        This method is run when calling the LogisticRegression class.
        :param tensor: the tensor that carries the input data (n_samples * 400)
        :return: an output tensor containing probability values for the output
        classes (n_samples * 4)
        """
        if self.value_num and self.embed_dim:
            embeds = self.embed(pp_head_data)
            b_size = embeds.size(0)
            pp_head_data = embeds.view(b_size, -1)

        out = self.linear(pp_head_data)
        return torch.sigmoid(out)


class NeuralCandidateScoringModel(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, value_num: int = 0, embed_dim: int = 0
    ):
        """
        A simple Feed-Forward model with one hidden layer, which classifies whether
        dependency heads for PPs are correct or not.
        :param input_dim: number of features
        :param output_dim: number of output classes
        """
        super(NeuralCandidateScoringModel, self).__init__()
        self.value_num = value_num
        self.embed_dim = embed_dim

        # For embedding without pretrained embeddings.
        if self.value_num and self.embed_dim:
            self.embed = torch.nn.Embedding(value_num, embed_dim)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim * embed_dim if embed_dim else input_dim, 256),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            torch.nn.Sigmoid(),
        )

    def forward(self, pp_head_data: FloatTensor):
        """
        Do the forward pass of the model.
        This method is run when calling the LogisticRegression class.
        :param tensor: the tensor that carries the input data (n_samples * 400)
        :return: an output tensor containing probability values for the output
        classes (n_samples * 4)
        """
        if self.value_num and self.embed_dim:
            embeds = self.embed(pp_head_data)
            b_size = embeds.size(0)
            pp_head_data = embeds.view(b_size, -1)
        out = self.model(pp_head_data)
        return out
