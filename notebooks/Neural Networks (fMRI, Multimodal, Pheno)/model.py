import time
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf

def build_model(input_dim, units, l2_reg, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(features, labels, hyperparameters, model_filename):
    units = hyperparameters['units']
    l2_reg = hyperparameters['l2_reg']
    dropout_rate = hyperparameters['dropout_rate']
    learning_rate = hyperparameters['learning_rate']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_conf_matrices = []
    fold_roc_curves = []
    fold_auc_scores = []

    best_model = None
    best_accuracy = 0

    start_time = time.time()  # start time

    for train_index, test_index in kfold.split(features, labels):
        X_train_fold, X_test_fold = features[train_index], features[test_index]
        y_train_fold, y_test_fold = labels.values[train_index], labels.values[test_index]

        y_train_fold = to_categorical(y_train_fold - 1, num_classes=2)
        y_test_fold = to_categorical(y_test_fold - 1, num_classes=2)

        model = build_model(X_train_fold.shape[1], units, l2_reg, dropout_rate, learning_rate)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_test_fold, y_test_fold),
            callbacks=[early_stopping], verbose=0
        )

        scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
        fold_accuracies.append(scores[1] * 100)

        if scores[1] > best_accuracy:
            best_accuracy = scores[1]
            best_model = model
            best_model.save(model_filename)

        # Predictions
        y_pred_prob = model.predict(X_test_fold)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_fold, axis=1)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        fold_conf_matrices.append(conf_matrix)

        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fold_roc_curves.append((fpr, tpr))
        fold_auc_scores.append(roc_auc)

    end_time = time.time()  # end time
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean accuracy over 5 folds: {mean_accuracy:.2f}%")
    print(f"Standard deviation over 5 folds: {std_accuracy:.2f}%")

    return fold_accuracies, fold_conf_matrices, fold_roc_curves, fold_auc_scores, history
