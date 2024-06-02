import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time


## MS2
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_size):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.model_NN = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        out = self.model_NN(x)
        preds: torch.Tensor = out

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
            nn.ReLU(),
            nn.Linear(n_classes, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.model_CNN(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        preds: torch.Tensor = x

        return preds


# Helper classes for the ViT
class PatchEmbedding(nn.Module):
    def __init__(self, chw, n_patches, hidden_d):
        super().__init__()
        self.ch, self.h, self.w = chw # channels, height, width
        self.patch_size = self.h // int(n_patches**0.5) # Assuming square patches
        self.n_patches = n_patches # Number of patches
        self.linear = nn.Linear(self.ch * self.patch_size * self.patch_size, hidden_d) # Linear layer to embed patches into hidden_d

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold( # Unfold the image into patches
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, C, self.n_patches, -1)  # Reshape the patches
        patches = patches.permute(0, 2, 1, 3).flatten(2) 
        embeddings = self.linear(patches)
        return embeddings


def positional_encoding(sequence_length, hidden_d, device):
    position = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1).to(device) # 0, 1, 2, ..., sequence_length
    div_term = torch.exp(torch.arange(0, hidden_d, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_d)).to(device) # 1/10000^(2i/d)
    
    pos_embeddings = torch.zeros(sequence_length, hidden_d).to(device)
    pos_embeddings[:, 0::2] = torch.sin(position * div_term) # add the sin to even indices
    pos_embeddings[:, 1::2] = torch.cos(position * div_term) # add the cos to odd indices

    return pos_embeddings.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_d, n_heads)
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, 4 * hidden_d),
            nn.GELU(),
            nn.Linear(4 * hidden_d, hidden_d),
        )
        self.norm2 = nn.LayerNorm(hidden_d)

    def forward(self, x):
        # pre layer normalization
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out # Residual connection

        # post layer normalization
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out # Residual connection
        return x

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d, device):
        """
        Initialize the network.

        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(chw, n_patches, hidden_d)
        self.positional_encoding = positional_encoding(n_patches, hidden_d, device)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(hidden_d, out_d)
        self.device = device

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.to(self.device)
        x = self.patch_embedding(x)
        x += self.positional_encoding.to(self.device)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)  # Global average pooling
        preds = self.fc(x)
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        start_time = time.time()
        for ep in range(self.epochs):
            epoch_start_time = time.time()
            self.train_one_epoch(dataloader, ep)
            epoch_end_time = time.time()
            
            elapsed_time = epoch_end_time - epoch_start_time
            total_elapsed_time = epoch_end_time - start_time
            epochs_left = self.epochs - (ep + 1)
            estimated_time_left = epochs_left * elapsed_time

            if(ep % 5 == 0):
                self.predict_torch(dataloader)

            print(f"Epoch [{ep+1}/{self.epochs}] completed.")
            print(f"Time for this epoch: {elapsed_time:.2f} seconds")
            print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
            print(f"Estimated time left: {estimated_time_left:.2f} seconds\n")

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(dataloader):
            print(f"Batch [{batch_idx+1}/{len(dataloader)}]", end="\r")

            data, targets = data.to(self.model.device), targets.to(self.model.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)

            # Compute the loss
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()

            # Update the model parameters
            self.optimizer.step()

            running_loss += loss.item()
            


        print(f"Epoch [{ep+1}] Loss: {running_loss/len(dataloader):.4f}")

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for data in dataloader:
                data = data[0].to(self.model.device)  # data[0] to get the input, ignore the target if exists
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)

                pred_labels.append(predicted)

        pred_labels = torch.cat(pred_labels)
        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        training_data = training_data.reshape(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
        print(training_data.shape)

        training_labels = training_labels.reshape(-1)  # Ensure labels are of shape (N,)
        print(training_labels.shape)

        
        # First, prepare data for pytorch
        train_dataset = TensorDataset(
            torch.from_numpy(training_data).float().to(self.device), torch.from_numpy(training_labels).long().to(self.device)
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        test_data = test_data.reshape(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)

        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float().to(self.device))
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
