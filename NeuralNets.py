import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pickle
import torch
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt
import torchsample
import torchsample.callbacks
import torchsample.metrics
import math
from torch.autograd import Variable
import pdfkit


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
        # false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
        # roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
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
        #  self.fc2 = torch.nn.Linear(hidden_size,
        #                         64)  # 1st Full-Connected Layer: 3072 (input data) -> 1024 (hidden node)
        self.fc2 = torch.nn.Linear(hidden_size,
                                   num_classes)  # 2nd Full-Connected Layer: 1023 (hidden node) -> 10 (output class)

    def forward(self, x):
        out = F.rrelu(self.fc1(x))  # followed by sigmoid activation function
        out = F.rrelu(self.fc2(out))  # followed by sigmoid activation function

        #  out = F.rrelu(self.fc3(out))

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
    # x = x[~np.isnan(x)]
    # Center to the mean and component wise scale to unit variance.
    x_scaled = sklearn.preprocessing.scale(x, with_mean=False)
    "Uncommented this line for taking out h2h feature"
    # last_x = np.column_stack((x_scaled, h2h))  # Add H2H statistics back to the mix

    # We need to get rid of the duplicate values in our dataset. (Around 600 dups. Will check it later)
    x_scaled_no_duplicates, indices = np.unique(x, axis=0, return_index=True)
    print(len)
    y_no_duplicates = y[indices]
    return [x_scaled_no_duplicates, y_no_duplicates, standard_deviations]


# Prepares our sets in tensor format
def prepare_train_validation_and_test_sets(features, labels, dev_set_size):
    # Specify the training dataset
    x_scaled_no_duplicates, y_no_duplicates, standard_deviations = preprocess_features_before_training(features,
                                                                                                       labels)
    x, x_test, y, y_test = sklearn.model_selection.train_test_split(x_scaled_no_duplicates, y_no_duplicates,
                                                                    test_size=0.2,
                                                                    shuffle=False)

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x, y, test_size=dev_set_size,
                                                                              shuffle=False)

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


def exp_decay(epoch, initial_rate):
    k = 0.8
    lrate = initial_rate * math.exp(-k * epoch)
    return lrate


def step_decay(epoch, initial_rate):
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_rate * math.pow(drop,
                                    math.floor((1 + epoch) / epochs_drop))
    return lrate


# To switch from Variable into numpy for testing
def resize_label_arrays(y_pred_torch, y):
    prob = F.softmax(y_pred_torch, dim=1)
    y_pred_np = y_pred_torch.data.numpy()
    pred_np = np.argmax(y_pred_np, axis=1)
    pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))
    label_np = y.numpy().reshape(len(y), 1)
    return pred_np, label_np


# To resize when we are making predictions on unseen data
def resize_prediction_arrays(y_pred_torch, y):
    #   prob = F.softmax(y_pred_torch, dim=1)
    #  print(prob)
    y_pred_np = y_pred_torch.data.numpy()
    pred_np = np.argmax(y_pred_np, axis=1)
    pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))
    label_np = y.reshape(len(y), 1)
    return pred_np, label_np


def calculate_roc_score(y, y_pred, train):
    false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(y, y_pred)
    roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
    print("ROC SCORE for " + train + " is : {}".format(roc_auc))


def make_nn_predictions(filename, tournament_pickle_file_name, x_scaled_no_duplicates, y_no_duplicates,
                        features_from_prediction_final, final_results,
                        match_to_results_dictionary, match_to_odds_dictionary, players, match_uncertainty_dict):
    bet_amount = 10
    total_winnings = 0
    count = 0
    correct = 0
    tournament_name = "wimbledon_2018_results.txt\n"
    f = open(tournament_name, "w+")
    match_to_results_list = list(match_to_results_dictionary.items())  # get list of matches

    linear_clf = Net(7, 128, 2)
    linear_clf.load_state_dict(torch.load(filename))  # 'ckpt.pth02.tar'

    y_pred, y_test = resize_prediction_arrays(
        linear_clf(Variable(torch.from_numpy(x_scaled_no_duplicates).float())), y_no_duplicates)
    print("Training accuracy {}\n".format(float(np.count_nonzero(y_pred == y_test)) / float(len(y_test))))
    calculate_roc_score(y_test, y_pred, 'train')

    f.write(tournament_name)
    f.write("Training accuracy {}\n".format(float(np.count_nonzero(y_pred == y_test)) / float(len(y_test))))

    y_pred, y_test = resize_prediction_arrays(
        linear_clf(Variable(torch.from_numpy(features_from_prediction_final).float())), final_results)
    print("Prediction accuracy {}".format(float(np.count_nonzero(y_pred == y_test)) / float(len(y_test))))
    f.write("Prediction accuracy {}\n".format(float(np.count_nonzero(y_pred == y_test)) / float(len(y_test))))

    calculate_roc_score(y_test, y_pred, 'test')

    for i, (feature, result) in enumerate(zip(features_from_prediction_final, final_results)):
        # USING SIMPLE FEED FORWARD NEURAL NETWORK
        # print(feature.shape)
        # print(feature.reshape(1, -1).shape)
        class_predictions = linear_clf(Variable(torch.from_numpy(feature).float()))
        # print(class_predictions)
        # print(class_predictions.shape)
        prob = F.softmax(class_predictions, dim=0)

        # print(type(prob.data.numpy()))
        prediction_probability = prob.data.numpy()[::-1]

        prediction = np.argmax(prob.data.numpy())

        calculated_odds_of_p1 = float(100 / prediction_probability[0] / 100)
        calculated_odds_of_p2 = float(100 / prediction_probability[1] / 100)

        match = match_to_results_list[i][0]  # can do this because everything is ordered
        odds = match_to_odds_dictionary[tuple(match)]
        uncertainty = match_uncertainty_dict[tuple(match)]
        total_bookmaker_prob = (1 / float(odds[0])) + (1 / float(odds[1]))
        bias = abs(1 - total_bookmaker_prob) / 2

        # predictions_dict[tuple(match)] = [prediction, result, np.asarray(prediction_probability),
        #                                 odds, odds[abs(int(prediction) - 1)]]
        if result == prediction:

            if abs(float(odds[abs(int(prediction) - 1)])) < 20:
                correct = correct + 1
                count = count + 1
                total_winnings = total_winnings + (
                        bet_amount * float(odds[abs(int(prediction) - 1)]))
        else:
            count = count + 1
            total_winnings = total_winnings - bet_amount

        # print("Our total winnings so far is {}".format(total_winnings))
        player1 = players[players['ID_P'] == list(match)[0]].iloc[0]['NAME_P']
        player2 = players[players['ID_P'] == list(match)[1]].iloc[0]['NAME_P']
        f.write("MATCH NUMBER {}. Uncertainty: {}\n".format(i, uncertainty))
        if (uncertainty < 0.2):
            f.write("This uncertainty means the quality of match features is GOOD.\n".format(i, uncertainty))
        else:
            f.write("This uncertainty means the quality of match features is NOT GOOD.\n".format(i, uncertainty))

        match_analysis = "Prediction for match {} - {} was {}. The result was {}.".format(
            player1, player2, prediction, result)
        our_probabilities = "The calculated model probabilities: \n{} : {:.2f} and {} : {:.2f}.".format(player1,
                                                                                                        prediction_probability[
                                                                                                            0],
                                                                                                        player2,
                                                                                                        prediction_probability[
                                                                                                            1])
        our_decimal_odds = "Converting our implied probability into decimal odds, model calculates odds as: \n{} :" \
                           " {:.2f} and {} : {:.2f}. ".format(player1, calculated_odds_of_p1, player2,
                                                              calculated_odds_of_p2)
        print(match_analysis)
        print(our_probabilities)
        print(our_decimal_odds)
        bookmaker_odds = "The bookmakers (actual) odds were: \n{} : {} and {} : {}".format(
            player1, odds[0], player2, odds[1])

        print(bookmaker_odds)
        bookmaker_probabilities = "The bookmaker probabilities  are: \n{} : " \
                                  "{:.2f} and {} : {:.2f}.".format(player1, (1 / (float(odds[0])) - bias), player2,
                                                                   (1 / (float(odds[1])) - bias))

        print(bookmaker_probabilities)

        odds_chosen = "The odds we chose to bet was {}".format(odds[abs(int(prediction) - 1)])

        print(odds_chosen)
        # Write the results to a text file
        f.write(match_analysis + "\n")
        f.write(our_decimal_odds + "\n")
        f.write(bookmaker_odds + "\n")
        f.write(our_probabilities + "\n")
        f.write(bookmaker_probabilities + "\n")
        f.write(odds_chosen + "\n")

    f.write("Total amount of bets we made is: {}".format(bet_amount * count))
    f.write("Total Winnings: {}".format(total_winnings))
    ROI = (total_winnings - (bet_amount * count)) / (bet_amount * count) * 100
    f.write("Our ROI for {} was: {}.".format(tournament_pickle_file_name, ROI))
    f.write("Accuracy over max probability and with odd threshold is {}.".format(correct / count))

    # Convert text file to pdf

    return linear_clf


class NeuralNetModel(object):

    def __init__(self, dataset_name, labelset_name, batchsize, dev_set_size, threshold, text_file=True):
        self.batchsize = batchsize
        self.threshold = threshold  # used for writing name of model file
        if text_file:
            pickle_in = open(dataset_name, "rb")
            features = np.asarray(pickle.load(pickle_in))
            pickle_in_2 = open(labelset_name, "rb")
            labels = np.asarray(pickle.load(pickle_in_2))
            # print(features.shape)
            # print(labels.shape)
            self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = prepare_train_validation_and_test_sets(
                features, labels, dev_set_size)
        # print(self.x_train.shape)
        # print(self.y_train.shape)

        else:
            # print(dataset_name.shape)
            # print(labelset_name.shape)

            self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = prepare_train_validation_and_test_sets(
                dataset_name, labelset_name, dev_set_size)
        # print(self.x_train.shape)
        # print(self.y_train.shape)

        self.lrs = []
        self.epochs = []
        self.val_losses = []
        self.train_nn_net_with_torchsample(7, self.batchsize)

    def early_stopping_schedule(self, epoch, lr, current_val_loss):
        self.lrs.append(lr)
        self.epochs.append(epoch)
        self.val_losses.append(current_val_loss)

    def lr_schedule(self, epoch, lr, current_val_loss):
        new_lr = exp_decay(epoch, lr[0])
        self.lrs.append(new_lr)
        self.epochs.append(epoch)
        self.val_losses.append(current_val_loss)
        return new_lr

    def clr_schedule(self, epoch, lr):
        # """ CYLICAL LEARNING RATE IMPLEMENTATION"
        print('\nCurrent learning weight is {}'.format(lr))
        self.lrs.append(lr)
        self.epochs.append(epoch)

    def train_nn_net_with_torchsample(self, feature_size, batch_size):
        model = Net(feature_size, 128, 2)
        # lr=0.1, momentum=0.55
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.8, nesterov=True)

        # optimizer = optim.Adam(model.parameters(), lr=0.00005) # good alternative for threshold = 0.1
        # optimizer = optim.Adam(model.parameters(), lr=0.00015) # better alternative for threshold = 0.15
        # optimizer = optim.Adagrad(model.parameters(), lr=0.0025)  # better alternative for threshold = 0.15
        # optimizer = optim.Adagrad(model.parameters(), lr=0.007)  # better alternative for threshold = 0.1
        # optimizer = optim.Adagrad(model.parameters(), lr=0.003)  # better alternative for threshold = 0.4 and batchsize = 128
        # optimizer = optim.Adam(model.parameters(), lr=0.00005) # better alternative for threshold = 0.4 and batchsize = 128

        criterion = torch.nn.CrossEntropyLoss()

        callbacks = [  # torchsample.callbacks.CyclicLR(step_size=500,base_lr = 0.001,max_lr = 0.01),
            torchsample.callbacks.EarlyStopping(schedule=self.early_stopping_schedule,
                                                monitor='val_loss',  # monitor loss
                                                min_delta=1e-5,
                                                patience=80),
            torchsample.callbacks.ModelCheckpoint(directory='~aysekozlu/PyCharmProjects/TennisModel',
                                                  monitor='val_loss', save_best_only=True, verbose=1)]
        # torchsample.callbacks.CyclicLR(self.early_stopping_schedule)]
        """
        
        torchsample.callbacks.EarlyStopping(schedule = self.early_stopping_schedule,
                                                         monitor='val_loss',  # monitor loss
                                                         min_delta=1e-5,
                                                         patience=80),
        [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.8,
                                                             patience=8,
                                                             cooldown=5,
                                                             epsilon=0.002,
                                                             min_lr=1e-8,
                                                             verbose=1),
        torchsample.callbacks.ModelCheckpoint(directory='~aysekozlu/PyCharmProjects/TennisModel',
                                                           monitor='val_loss', save_best_only=True, verbose=1)]
        torchsample.callbacks.LRScheduler(self.lr_schedule),
        [torchsample.callbacks.EarlyStopping(schedule = self.early_stopping_schedule,
                                                         monitor='val_loss',  # monitor loss
                                                         min_delta=1e-5,
                                                         patience=10)"""

        metrics = [Binary_Classification()]  # Keep track of validation and training accuracy each epoch
        trainer = torchsample.modules.ModuleTrainer(model)
        print(type(trainer))

        trainer.compile(loss=criterion,
                        optimizer=optimizer,
                        metrics=metrics)
        trainer.set_callbacks(callbacks)
        print(batch_size)
        trainer.fit(self.x_train, self.y_train,
                    val_data=(self.x_val, self.y_val),
                    num_epoch=200,
                    batch_size=batch_size,
                    verbose=1)

        trainer.evaluate(self.x_train, self.y_train)

        # Calculate training accuracy
        y_pred_train_np, y_train_np = resize_label_arrays(trainer.predict(self.x_train), self.y_train)
        print("Training accuracy {}".format(
            float(np.count_nonzero(y_pred_train_np == y_train_np)) / float(len(y_train_np))))
        calculate_roc_score(y_train_np, y_pred_train_np, 'train')
        # Calculate testing accuracy
        y_pred_test_np, y_test_np = resize_label_arrays(trainer.predict(self.x_test), self.y_test)
        print(
            "Testing accuracy {}".format(float(np.count_nonzero(y_pred_test_np == y_test_np)) / float(len(y_test_np))))
        calculate_roc_score(y_test_np, y_pred_test_np, 'test')

        plt.figure()
        plt.plot(self.epochs, self.lrs)
        plt.title('Learning rate over Epoches')
        plt.xticks(np.arange(0.7, 0.6, step=0.1))

        plt.xlabel('Number of Epoch')
        plt.ylabel('Learning Rate')

        plt.figure()
        plt.plot(self.epochs, self.val_losses)
        plt.title('Loss over Epoches')
        plt.ylabel('Validation Loss')
        plt.xticks(np.arange(0.7, 0.6, step=0.1))

        plt.xlabel('Number of Epoch')
        plt.show()


# NN = NeuralNetModel("data_v12_short.txt", "label_v12_short.txt", 128, 0.5)

"""
 optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.55)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.8,
                                                             patience=10,
                                                             cooldown=3,
                                                             epsilon=0.03,
                                                             min_lr=1e-8,
                                                             verbose=1)]
                                                             
results: 
Epoch 100/100: : 497 batches [00:00, 785.92 batches/s, val_loss=0.6298, val_acc=64.85, acc=65.08, loss=0.4700, lr=0.0210]                 
Training accuracy 0.6523274883310206
Testing accuracy 0.6534216335540839


optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.8)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.5,
                                                             patience=10,
                                                             cooldown=3,
                                                             epsilon=0,
                                                             min_lr=1e-8,
                                                             verbose=1)]
                                                             
                                                             batch size = 128 
                                                             Epoch 99/100: : 249 batches [00:00, 968.38 batches/s, val_loss=0.6286, val_acc=64.95, acc=64.86, loss=0.6067, lr=0.0250]                  
Epoch 100/100: : 249 batches [00:00, 946.21 batches/s, val_loss=0.6286, val_acc=64.97, acc=64.84, loss=0.6066, lr=0.0250]                  




Randomized Relu:
   model = Net(feature_size, 128, 2)
        # lr=0.1, momentum=0.55
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.5,
                                                             patience=8,
                                                             cooldown=1,
                                                             epsilon=0.001,
                                                             min_lr=1e-8,
                                                             verbose=1)]
OR - LEARNING RATE SLIGHTLY LESS  
optimizer = optim.SGD(model.parameters(), lr=0.0008, momentum=0.8, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.8,
                                                             patience=8,
                                                             cooldown=1,
                                                             epsilon=0.002,
                                                             min_lr=1e-8,
                                                             verbose=1)]                                          
Epoch 100/100: : 249 batches [00:00, 505.84 batches/s, val_loss=0.6334, val_acc=64.95, acc=65.26, loss=0.6241, lr=0.0030]                 
Training accuracy 0.6518228838148101
Testing accuracy 0.6529801324503312
batch size = 128 


When getting 0.2 of uncertainty: 
 optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.8, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss()
        callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                             , monitor='val_loss',
                                                             factor=0.8,
                                                             patience=8,
                                                             cooldown=1,
                                                             epsilon=0.002,
                                                             min_lr=1e-8,
                                                             verbose=1)]

When getting 0.1 of uncertainty: 
same true when another fc3 added of size 64
model = Net(feature_size, 128, 2)
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.8, nesterov=True)
criterion = torch.nn.CrossEntropyLoss()
callbacks = [torchsample.callbacks.ReduceLROnPlateau(schedule=self.early_stopping_schedule
                                                     , monitor='val_loss',
                                                     factor=0.8,
                                                     patience=8,
                                                     cooldown=5,
                                                     epsilon=0.002,
                                                     min_lr=1e-8,
                                                     verbose=1),
    """
