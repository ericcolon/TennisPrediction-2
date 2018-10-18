import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
import torch
import sklearn
import sklearn.preprocessing
import sklearn.model_selection

import torchsample
import torchsample.callbacks
import torchsample.metrics


# I Overwrite this class from torchsample. The one at torch sample does not work for multi class outputs.
class Binary_Classification(torchsample.metrics.BinaryAccuracy):
    def __call__(self, y_pred, y_true):
        y_pred = y_pred.data.numpy()  # Turn it into np arrays
        y_true = y_true.data.numpy()
        y_pred = np.argmax(y_pred, axis=1)  # get position of max probability

        y_pred = y_pred.reshape(len(y_pred), 1)
        y_true = y_true.reshape(len(y_true), 1)
        self.correct_count += np.count_nonzero(y_pred == y_true)
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


# Overwritten DataLoader class from Pytorch
class DS(Dataset):

    def __init__(self, x, y):
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# Pytorch Neural Net module is used to create the neural net.
class Net(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(input_size,
                                   hidden_size)  # 1st Full-Connected Layer: 3072 (input data) -> 1024 (hidden node)
        self.fc2 = torch.nn.Linear(hidden_size,
                                   num_classes)  # 2nd Full-Connected Layer: 1023 (hidden node) -> 10 (output class)

    def forward(self, x):
        out = F.relu(self.fc1(x))  # followed by sigmoid activation function
        out = F.relu(self.fc2(out))

        return out


def preprocess_features_before_training(features, labels):
    # 1. Reverse the feature and label set
    # 2. Scale the features to unit variance (except h2h feature). Also save the std. deviation of each feature
    # 3. Remove any duplicates (so we don't have any hard to find problems with dictionaries later
    x = features[::-1]
    y = labels[::-1]
    # number_of_columns = x.shape[1] - 1

    # Before standardizing we want to take out the H2H column
    # h2h = x[:, number_of_columns]

    # Delete this specific column
    x_shortened = np.delete(x, np.s_[-1], 1)

    standard_deviations = np.std(x, axis=0)

    # Center to the mean and component wise scale to unit variance.
    x_scaled = sklearn.preprocessing.scale(x, with_mean=False)
    "Uncommented this line for taking out h2h feature"
    # last_x = np.column_stack((x_scaled, h2h))  # Add H2H statistics back to the mix

    # We need to get rid of the duplicate values in our dataset. (Around 600 dups. Will check it later)
    x_scaled_no_duplicates, indices = np.unique(x, axis=0, return_index=True)

    y_no_duplicates = y[indices]
    return [x_scaled_no_duplicates, y_no_duplicates, standard_deviations]


# Prepares our sets in tensor format
def prepare_train_validation_and_test_sets(dataset_name, labelset_name):
    pickle_in = open(dataset_name, "rb")
    features = np.asarray(pickle.load(pickle_in))
    pickle_in_2 = open(labelset_name, "rb")
    labels = np.asarray(pickle.load(pickle_in_2))
    # Specify the training dataset
    x_scaled_no_duplicates, y_no_duplicates, standard_deviations = preprocess_features_before_training(features,
                                                                                                       labels)
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x_scaled_no_duplicates, y_no_duplicates,
                                                                    test_size=0.2,
                                                                    shuffle=True)

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=0.2,
                                                                              shuffle=True)

    train_dataset = DS(x_train, y_train)
    print("Length of train SET is {}".format(train_dataset.__len__()))
    # train_loader = DataLoader(dataset=train_dataset,
    #                          batch_size=batch_size, shuffle=True)

    test_dataset = DS(x_test, y_test)
    print("Length of test SET is {}".format(test_dataset.__len__()))

    # test_loader = DataLoader(dataset=test_dataset,
    #                         batch_size=batch_size, shuffle=True)

    val_dataset = DS(x_val, y_val)
    print("Length of validation SET is {}".format(val_dataset.__len__()))

    # val_loader = DataLoader(dataset=val_dataset,
    #         batch_size=batch_size, shuffle=True)

    return [train_dataset.x_data, train_dataset.y_data.long().squeeze(1), test_dataset.x_data,
            test_dataset.y_data.long().squeeze(1), val_dataset.x_data, val_dataset.y_data.long().squeeze(1)]
    # train_and_test_ff_network(train_loader, test_loader, x_test, y_test)


# To switch from Variable into numpy for testing
def resize_label_arrays(y_pred_torch, y):
    y_pred_np = y_pred_torch.data.numpy()
    pred_np = np.argmax(y_pred_np, axis=1)
    pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))
    label_np = y.numpy().reshape(len(y), 1)
    return pred_np, label_np


def calculate_roc_score(y, y_pred):
    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y, y_pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    print("ROC SCORE for training is : {}".format(roc_auc))


class NeuralNetModel(object):

    def __init__(self, dataset_name, labelset_name, batchsize):
        self.batchsize = batchsize
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = prepare_train_validation_and_test_sets(
            dataset_name, labelset_name)

    def train_nn_net_with_torchsample(self):
        model = Net(7, 64, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.55)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.EarlyStopping(monitor='val_loss',  # monitor loss
                                                         min_delta=0,
                                                         patience=10
                                                         )]#,torchsample.callbacks.LRScheduler(schedule=1)]

        metrics = [Binary_Classification()]  # Keep track of validation and training accuracy each epoch
        trainer = torchsample.modules.ModuleTrainer(model)
        print(type(trainer))

        trainer.compile(loss=criterion,
                        optimizer=optimizer,
                       metrics=metrics)
        trainer.set_callbacks(callbacks)

        trainer.fit(self.x_train, self.y_train,
                    val_data=(self.x_val, self.y_val),
                    num_epoch=250,
                    batch_size=64,
                    verbose=1)

        trainer.evaluate(self.x_train, self.y_train)


        # Calculate training accuracy
        y_pred_train_np, y_train_np = resize_label_arrays(trainer.predict(self.x_train), self.y_train)
        print("Training accuracy {}".format(
            float(np.count_nonzero(y_pred_train_np == y_train_np)) / float(len(y_train_np))))

        # Calculate testing accuracy
        y_pred_test_np, y_test_np = resize_label_arrays(trainer.predict(self.x_test), self.y_test)
        print(
            "Testing accuracy {}".format(float(np.count_nonzero(y_pred_test_np == y_test_np)) / float(len(y_test_np))))


NN = NeuralNetModel("data_v12_short.txt", "label_v12_short.txt", 64)
NN.train_nn_net_with_torchsample()
