import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def get_metrics(y_true, y_pred):
    # Assuming y_true and y_pred are your arrays of true labels and predictions respectively
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # ROC-AUC score calculation for binary classification
    # For multi-class, you need the prediction scores, not the predicted labels, and use a different strategy
    roc_auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) == 2 else "ROC-AUC not applicable"

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print the metrics
    print(f"Accuracy: {accuracy:.2f}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC-AUC Score: {roc_auc}")
    # print(f"Confusion Matrix:\n{conf_matrix}")

    return accuracy, roc_auc, f1

def dummy_vs_true_labels(data_loader):
    '''
    Compare the predictions of the dummy classifier with the ground truth labels (0 for left, 1 for right).

    Assuming the structure of the data_loader is known and each batch contains inputs, fixations, and labels
    We will define a function to process each batch and predict the turn direction based on the dummy assumption 
    that when the driver looks more on the right, they will turn right and respectivrly for the left.
    '''
    total_correct = 0
    total_samples = 0
        
    true_labels = []
    predictions = []

    for data in data_loader:
        # Unpack the data
        inputs, fixations, labels = data
        
        true_labels.extend(labels.tolist())

        # Keep only the first channel of the fixations
        fixations = fixations[:, 0, :, :]  # This selects the first channel for all images in the batch

        # Predict directions for the batch
        for fixation in fixations:
            left_half_sum = torch.sum(fixation[:, :fixation.shape[1] // 2])
            right_half_sum = torch.sum(fixation[:, fixation.shape[1] // 2:])
            prediction = 1 if right_half_sum > left_half_sum else 0
            predictions.append(prediction)

        # Calculate the number of correct predictions
        correct_predictions = sum(pred == true_labels[i] for i, pred in enumerate(predictions))

        # Update totals
        total_correct += correct_predictions
        total_samples += len(labels)

    # Calculate mean accuracy
    accuracy, auc, f1 = get_metrics(true_labels, predictions)
    # mean_accuracy = total_correct / total_samples
    return accuracy, auc, f1