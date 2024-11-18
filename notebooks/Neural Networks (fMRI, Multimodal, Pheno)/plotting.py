import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_confusion_matrix(conf_matrices, classes):
    plt.figure(figsize=(6, 6))
    mean_conf_matrix = np.mean(conf_matrices, axis=0)
    plt.imshow(mean_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Mean Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = mean_conf_matrix.max() / 2.
    for i in range(mean_conf_matrix.shape[0]):
        for j in range(mean_conf_matrix.shape[1]):
            plt.text(j, i, f'{mean_conf_matrix[i, j]:.0f}',
                     horizontalalignment='center',
                     color='white' if mean_conf_matrix[i, j] > thresh else 'black')
    plt.show()

def plot_roc_curves(roc_curves, auc_scores):
    plt.figure(figsize=(6, 6))
    for i, (fpr, tpr) in enumerate(roc_curves):
        plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {auc_scores[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
