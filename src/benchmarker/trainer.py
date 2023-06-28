import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from src.benchmarker.models import BertBinaryClassifier


def load_model(model_path):
    bert_clf = BertBinaryClassifier()
    bert_clf.load_state_dict(torch.load(model_path))
    bert_clf.cuda()
    return bert_clf


def data_loader(dataset, shuffle=False, batch_size=8):
    torch.cuda.empty_cache()
    CUDA_VISIBLE_DEVICES = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return dataloader


def train_val_split(data, validation_split, shuffle=True, random_seed=42):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return data.iloc[train_indices], data.iloc[val_indices]


def train(
    train_set,
    epochs,
    batch_size,
    shuffle,
    validation_split,
    learning_rate,
    random_seed,
):
    train_set, val_set = train_val_split(
        train_set, validation_split, shuffle=shuffle, random_seed=random_seed
    )
    train_dataloader = data_loader(train_set, shuffle=shuffle, batch_size=batch_size)
    validation_loader = data_loader(val_set, shuffle=shuffle, batch_size=batch_size)

    bert_clf = BertBinaryClassifier()
    bert_clf = bert_clf.cuda()
    optimizer = torch.optim.Adam(bert_clf.parameters(), lr=learning_rate)
    accuracy = 0
    best_model = bert_clf
    for epoch_num in range(epochs):
        bert_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            token_ids, masks, labels, IDs = tuple(t for t in batch_data)
            token_ids, masks, labels, IDs = (
                token_ids.cuda(),
                masks.cuda(),
                labels.cuda(),
                IDs.cuda(),
            )
            probas = bert_clf(token_ids, masks)
            loss_func = nn.BCELoss()
            batch_loss = loss_func(probas, labels)
            train_loss += batch_loss.item()
            bert_clf.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print("Epoch: ", epoch_num + 1)
            print(
                "\r"
                + "{0}/{1} loss: {2} ".format(
                    step_num, len(train_set) / batch_size, train_loss / (step_num + 1)
                )
            )
        # save the model only if the accuracy on the test set was improved
        bert_clf.eval()
        predictions = []
        for step_num, batch_data in enumerate(validation_loader):
            token_ids, masks, true_label, IDs = tuple(t for t in batch_data)
            token_ids, masks, labels, IDs = (
                token_ids.cuda(),
                masks.cuda(),
                labels.cuda(),
                IDs.cuda(),
            )
            probs = bert_clf(token_ids, masks)
            probs = probs.detach().cpu().numpy()[0][0]
            if probs < 0.5:
                pred_label = 0
            else:
                pred_label = 1
            predictions.append(pred_label == true_label.cpu().numpy())
        if sum(predictions) / len(predictions) > accuracy:
            # update accuracy and save the model
            accuracy = sum(predictions) / len(predictions)
            print("accuracy was improved")
            print("accuracy= {}".format(accuracy))
            # torch.save(bert_clf.state_dict(), "bert_clf.pt")
            best_model = bert_clf

        else:
            print("accuracy was not improved")
    return best_model


def predict(model, test_set, batch_size):
    test_dataloader = data_loader(test_set, shuffle=False, batch_size=batch_size)
    model.eval()
    pred = pd.DataFrame(
        {"id": np.zeros(len(test_set)), "is_fake": np.zeros(len(test_set))}
    )
    for step_num, batch_data in enumerate(test_dataloader):
        token_ids, masks, IDs = tuple(t for t in batch_data)
        IDs = IDs.numpy()[0]
        token_ids, masks = token_ids.cuda(), masks.cuda()
        probs = model(token_ids, masks)
        probs = probs.detach().cpu().numpy()[0][0]
        pred.loc[step_num, "id"] = IDs
        if probs < 0.5:
            pred.loc[step_num, "is_fake"] = 0
        else:
            pred.loc[step_num, "is_fake"] = 1
        print(step_num)

    pred["id"] = pred["id"].astype("int32")
    pred["is_fake"] = pred["is_fake"].astype("int32")
    return pred
