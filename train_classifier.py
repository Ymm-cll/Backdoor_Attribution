import json
import os
import pickle
import random
import util
from baukit.baukit import TraceDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import get_ds

sns.set(style="whitegrid")


def format_input(input, tokenizer):
    if isinstance(input, str):
        input = [{"role": "user", "content": input}]
        input = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=False)
    if isinstance(input, list):
        for i, item in enumerate(input):
            item = [{"role": "user", "content": item}]
            item = tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=False)
            input[i] = item
    return input


def get_last_token_hidden_states(model, tokenizer, batch, max_length=256):
    tokenizer.padding_side = 'left'
    batch = format_input(batch, tokenizer)
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).to(model.device)

    num_hidden_layers = model.config.num_hidden_layers

    target_layers = [f'model.layers.{layer}.self_attn.q_proj' for layer in range(num_hidden_layers)]
    hidden_states = [[] for _ in range(model.config.num_hidden_layers)]
    with TraceDict(model, layers=target_layers, retain_input=True) as td:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            for i, layer in enumerate(target_layers):
                layer_hidden_state = td[layer].input  # batch_size * seq_length * hidden_dim
                last_token_layer_hidden_states = layer_hidden_state[
                    torch.arange(len(batch)), -1]  # batch_size *  hidden_dim
                hidden_states[i] += torch.unbind(last_token_layer_hidden_states.cpu().float(), dim=0)  # hidden_dim
    return hidden_states


def collect(data, batch_size, model, tokenizer):
    hidden_states = [[] for _ in range(model.config.num_hidden_layers)]
    for i in tqdm(range(0, len(data), batch_size), desc="collecting hidden states"):
        batch = data[i:i + batch_size]
        batch_hidden_states = get_last_token_hidden_states(model, tokenizer, batch)
        for i, layer_batch_hidden_states in enumerate(batch_hidden_states):
            hidden_states[i] += layer_batch_hidden_states
    return hidden_states


def get_classification_ds(task_name, trigger, model, tokenizer, batch_size=32):
    clean_data = json.load(open(f"./dataset/{task_name}/all.json", "r"))
    clean_data = [item["input"] for item in clean_data]
    add_trigger = get_ds.get_add_trigger(task_name)
    poison_data = [add_trigger(item, trigger) for item in clean_data]

    negative_set = collect(clean_data, batch_size, model, tokenizer)
    positive_set = collect(poison_data, batch_size, model, tokenizer)
    return negative_set, positive_set


class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, output_dim=1):
        super(EnhancedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train_mlp_classifier(positive_samples, negative_samples, hidden_dim, epochs=100, batch_size=16, lr=0.001):
    pos_tensors = torch.stack(positive_samples)
    neg_tensors = torch.stack(negative_samples)

    pos_labels = torch.ones(len(pos_tensors), 1)
    neg_labels = torch.zeros(len(neg_tensors), 1)

    X = torch.cat([pos_tensors, neg_tensors])
    y = torch.cat([pos_labels, neg_labels])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedMLP(input_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model = None
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            best_epoch = epoch

    model.load_state_dict(best_model)

    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    model.eval()
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    final_train_accuracy = train_correct / train_total
    final_val_accuracy = val_correct / val_total

    return {
        'model': model,
        'train_accuracy': final_train_accuracy,
        'test_accuracy': test_accuracy,
        'best_epoch': best_epoch,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'val_accuracy': final_val_accuracy,
    }


def train_svm_classifier(positive_samples, negative_samples, hidden_dim, C=1.0, gamma='scale', kernel='rbf'):
    pos_tensors = torch.stack(positive_samples).numpy()
    neg_tensors = torch.stack(negative_samples).numpy()

    pos_labels = np.ones(len(pos_tensors))
    neg_labels = np.zeros(len(neg_tensors))

    X = np.vstack([pos_tensors, neg_tensors])
    y = np.hstack([pos_labels, neg_labels])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    return {
        'model': model,
        'scaler': scaler,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'best_epoch': None,  # SVM doesn't use epochs
        'train_accuracies': [train_accuracy],
        'val_accuracies': [val_accuracy],
    }


def train_weak_classifier(positive_samples, negative_samples, hidden_dim, classifier_type='mlp', **kwargs):
    if classifier_type.lower() == 'mlp':
        return train_mlp_classifier(positive_samples, negative_samples, hidden_dim, **kwargs)
    elif classifier_type.lower() == 'svm':
        return train_svm_classifier(positive_samples, negative_samples, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}. Choose 'mlp' or 'svm'.")


def train_all_classifiers(positive_set, negative_set, hidden_dim, num_layers, classifier_type='mlp', **kwargs):
    classifiers = []
    results = []

    for i in tqdm(range(num_layers), desc=f"Training {classifier_type.upper()} weak classifiers"):
        pos_samples = positive_set[i]
        neg_samples = negative_set[i]

        result = train_weak_classifier(pos_samples, neg_samples, hidden_dim, classifier_type, **kwargs)
        classifiers.append(result['model'])
        if classifier_type.lower() == 'svm':
            classifiers[-1] = {'model': result['model'], 'scaler': result['scaler']}

        results.append({
            'classifier_idx': i,
            'train_accuracy': result['train_accuracy'],
            'val_accuracy': result['val_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'best_epoch': result['best_epoch']
        })
        res = results[-1]
        print(
            f"{res['classifier_idx']}\t{res['train_accuracy']:.4f}\t{res['val_accuracy']:.4f}\t\t{res['test_accuracy']:.4f}\t\t{res['best_epoch']}")

    return classifiers, results


def evaluate_cross_test_sets(classifiers, positive_set, negative_set, classifier_type='mlp'):
    num_classifiers = len(classifiers)
    cross_accuracies = np.zeros((num_classifiers, num_classifiers))

    test_sets = []
    for i in range(num_classifiers):
        pos_tensors = torch.stack(positive_set[i])
        neg_tensors = torch.stack(negative_set[i])

        pos_labels = torch.ones(len(pos_tensors), 1) if classifier_type.lower() == 'mlp' else np.ones(len(pos_tensors))
        neg_labels = torch.zeros(len(neg_tensors), 1) if classifier_type.lower() == 'mlp' else np.zeros(
            len(neg_tensors))

        if classifier_type.lower() == 'mlp':
            X = torch.cat([pos_tensors, neg_tensors])
            y = torch.cat([pos_labels, neg_labels])
        else:
            X = np.vstack([pos_tensors.numpy(), neg_tensors.numpy()])
            y = np.hstack([pos_labels, neg_labels])

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        test_sets.append((X_test, y_test))

    for i in tqdm(range(num_classifiers), desc="Evaluating cross-test accuracy"):
        if classifier_type.lower() == 'mlp':
            model = classifiers[i]
            model.eval()
        else:
            model_info = classifiers[i]
            model = model_info['model']
            scaler = model_info['scaler']

        for j in range(num_classifiers):
            X_test, y_test = test_sets[j]

            if classifier_type.lower() == 'mlp':
                test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8, shuffle=False)
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / total
            else:
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

            cross_accuracies[i, j] = accuracy

    return cross_accuracies


def plot_cross_accuracy_matrix(cross_accuracies, classifier_type='mlp'):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cross_accuracies, annot=False, fmt=".2f", cmap="YlGnBu",
                xticklabels=range(1, len(cross_accuracies) + 1),
                yticklabels=range(1, len(cross_accuracies) + 1))
    plt.title(f"{classifier_type.upper()} Classifier Cross-Test Accuracy Matrix")
    plt.xlabel("Test Set (Classifier #)")
    plt.ylabel("Classifier")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = "cuda:2"
    task_names = ["agnews_sentence"]
    model_families = ["qwen2.5-7b", "llama2-7b"]
    batch_size = 32
    classifier_types = ["svm", "mlp"]

    for task_name in task_names:
        for model_family in model_families:
            for classifier_type in classifier_types:
                trigger = None
                if task_name == "alpaca_begin":
                    trigger = "Current year: 2024."
                if task_name == "harmful_random":
                    trigger = "SUDO"

                base_model_path = None
                if model_family == "qwen2.5-7b":
                    base_model_path = util.qwen_model_path
                if model_family == "llama2-7b":
                    base_model_path = util.llama_model_path
                lora_model_path = f"./model_weight/{task_name}/{model_family}/attn_mlp/checkpoint-560"

                model, tokenizer = util.get_mt(base_model_path, device, lora_model_path)
                model = model.merge_and_unload()
                hidden_dim = model.config.hidden_size
                num_layers = model.config.num_hidden_layers

                negative_set, positive_set = get_classification_ds(task_name, trigger, model, tokenizer)
                del model
                del tokenizer

                classifier_params = {}
                if classifier_type.lower() == 'mlp':
                    classifier_params = {
                        'epochs': 100,
                        'batch_size': 16,
                        'lr': 0.001
                    }
                elif classifier_type.lower() == 'svm':
                    classifier_params = {
                        'C': 1.0,
                        'gamma': 'scale',
                        'kernel': 'rbf'
                    }

                print(f"Training {classifier_type.upper()} classifiers...")
                trained_classifiers, training_results = train_all_classifiers(
                    positive_set, negative_set, hidden_dim, num_layers,
                    classifier_type=classifier_type, **classifier_params
                )

                cross_accuracies = evaluate_cross_test_sets(
                    trained_classifiers, positive_set, negative_set, classifier_type
                )

                print("\nOriginal Test Accuracies:")
                for i in range(num_layers):
                    print(f"Classifier {i + 1}: {cross_accuracies[i, i]:.4f}")

                non_diag = cross_accuracies[~np.eye(num_layers, dtype=bool)]
                print(f"\nAverage Cross-Test Accuracy (excluding diagonal): {non_diag.mean():.4f}")

                os.makedirs(f"./results/probe/{classifier_type}/{task_name}/{model_family}", exist_ok=True)
                suffix = f"_{classifier_type.lower()}"
                with open(f"./results/probe/{classifier_type}/{task_name}/{model_family}/acc{suffix}.pkl", "wb") as f:
                    pickle.dump(training_results, f)
                with open(f"./results/probe/{classifier_type}/{task_name}/{model_family}/cross_acc{suffix}.pkl",
                          "wb") as f:
                    pickle.dump(cross_accuracies, f)
                plot_cross_accuracy_matrix(cross_accuracies, classifier_type)
