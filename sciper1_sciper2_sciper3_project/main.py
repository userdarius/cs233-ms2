import argparse

import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import torch

import torch

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, ResNet, Trainer, MyViT, CNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    print(f"Loading data from '{args.data}'...")
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    means = np.mean(xtrain, axis=0, keepdims=True)
    stds = np.std(xtrain, axis=0, keepdims=True)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    # Make a validation set
    if not args.test:
        print("Using a validation set")
        N = xtrain.shape[0]
        Nvalid = int(0.2 * N)
        xtrain, xvalid = xtrain[:-Nvalid], xtrain[-Nvalid:]
        ytrain, yvalid = ytrain[:-Nvalid], ytrain[-Nvalid:]
    else:
        print("Using PCA")
        xtrain = np.concatenate([xtrain, xtest], axis=0)
        ytrain = np.concatenate([ytrain, np.zeros(xtest.shape[0])], axis=0)

    ### WRITE YOUR CODE HERE to do any other data processing

    # Move data to MPS device and convert to float32
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Dimensionality reduction (MS2)
    best_explained_variance = 0
    best_pca_d = None
    if args.use_pca:
        print("Evaluating PCA dimensions...")
        exvar_values = []
        component_values = range(0, 600)
        for pca_d in component_values:
            pca_obj = PCA(d=pca_d)
            explained_variance = pca_obj.find_principal_components(xtrain)
            exvar_values.append(explained_variance)
            print(f"PCA d = {pca_d}, Explained Variance = {explained_variance:.2f}%")
            if explained_variance > best_explained_variance:
                best_explained_variance = explained_variance
                best_pca_d = pca_d

        # Plot the Explained Variance vs. Number of Components
        plt.figure()
        plt.plot(component_values, exvar_values)
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.savefig("plots/pca.jpg")

        print(f"Best PCA dimension: {best_pca_d} with explained variance: {best_explained_variance:.2f}%")
        pca_obj = PCA(d=best_pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        if not args.test:
            xvalid = pca_obj.reduce_dimension(xvalid)
        xtest = pca_obj.reduce_dimension(xtest)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    print(f"Number of classes: {n_classes}")
    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes, device=device)
    elif args.nn_type == "alexnet":
        model = CNN(input_channels=1, n_classes=n_classes, device=device)
    elif args.nn_type == "resnet":
        model = ResNet([3, 4, 6, 3], 10, num_channels=1)
    elif args.nn_type == "transformer":
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=12, hidden_d=256, n_heads=8, out_d=10, device=device)

    # Trainer object
    if args.exp_sch:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device,
                            use_exp_lr=True, exp_lr_params={"gamma": args.gamma})
    else:
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device)

    ## 4. Train and evaluate the method
    print(f"\nTraining {args.nn_type} model...")
    print(f"Using device: {device}")

    # Fit (:=train) the method on the training data
    preds_train, logs = method_obj.fit(xtrain, ytrain)
    # Predict on unseen data
    preds = method_obj.predict(xvalid)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, yvalid)
    macrof1 = macrof1_fn(preds, yvalid)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="../dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=16, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--exp_sch', action="store_true", help="use exponential lr")
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")

    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate for methods with learning rate")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
