import torch
import random
import numpy as np
from tqdm import trange, tqdm
from layers import StackedGCN
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score


class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN
    """
    def __init__(self, args, clustering_machine):
        """
        :param args: Arguments object.
        :param clustering_machine:
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transfer to GPU/CPU
        """
        self.model = StackedGCN(self.args, self.clustering_machine.features_count, self.clustering_machine.target_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster:Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        predictions = self.model(edges, features)
        average_loss = F.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]

        return average_loss, node_count

    def do_forward_pass_batch(self, batch):
        """
        Making a forward pass with data from a given batch_partition.
        :param batch: Batch index.
        :return: average_loss: Average loss on the batch.
        :return: node_count: Number of nodes.
        """
        edges = self.clustering_machine.batch_edges[batch].to(self.device)
        macro_nodes = self.clustering_machine.batch_nodes[batch].to(self.device)
        train_nodes = self.clustering_machine.batch_train_nodes[batch].to(self.device)
        features = self.clustering_machine.batch_features[batch].to(self.device)
        target = self.clustering_machine.batch_targets[batch].to(self.device).squeeze()
        norm_eges = self.clustering_machine.norm_mx[batch].to(self.device)
        predictions = self.model(norm_eges, features)
        average_loss = F.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count  = train_nodes.shape[0]

        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count:Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item() * node_count

        self.node_count_seen = self.node_count_seen + node_count

        average_loss = self.accumulated_training_loss / self.node_count_seen

        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]

        prediction = self.model(edges, features)
        prediction = prediction[test_nodes, :]

        return prediction, target

    def do_prediction_batch(self, batch):
        """
        Scoring a batch.
        :param batch: batch: Batch index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.batch_edges[batch].to(self.device)
        macro_nodes = self.clustering_machine.batch_nodes[batch].to(self.device)
        test_nodes = self.clustering_machine.batch_test_nodes[batch].to(self.device)
        features = self.clustering_machine.batch_features[batch].to(self.device)
        target = self.clustering_machine.batch_targets[batch].to(self.device).squeeze()
        target = target[test_nodes]
        norm_eges = self.clustering_machine.norm_mx[batch].to(self.device)

        # predictions = self.model(edges, features)
        predictions = self.model(norm_eges, features)
        predictions = predictions[test_nodes, :]

        return predictions, target

    def train(self):
        """
        Training a model.
        """
        print("Training started.\n")
        epochs = trange(self.args.epochs, desc="Train loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        """
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            epochs.set_description("Train Loss: %g" % round(average_loss, 4))
        """
        for epoch in epochs:
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for batch in range(self.clustering_machine.batch_num):
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass_batch(batch)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            epochs.set_description("Train Loss: %g" % round(average_loss, 4))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        """
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        """
        for batch in range(self.clustering_machine.batch_num):
            prediction, target = self.do_prediction_batch(batch)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())

        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="micro")
        print("\nF-1 score: {:.4f}".format(score))








