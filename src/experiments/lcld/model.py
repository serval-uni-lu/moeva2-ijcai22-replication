from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense
import tensorflow as tf


def create_model():
    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
    )
    return model


def train_model(x_train_s, y_train):
    x_train_local, x_val_local, y_train_local, y_val_local = train_test_split(
        x_train_s,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )
    model = create_model()
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
    model.fit(
        x=x_train_local,
        y=y_train_local,
        epochs=100,
        batch_size=512,
        validation_data=(x_val_local, y_val_local),
        verbose=1,
        callbacks=[early_stop],
    )
    return model


def print_score(label, prediction):
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print("Classification Report:", end="")
    print(f"\tPrecision Score: {precision_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tRecall Score: {recall_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tF1 score: {f1_score(label, prediction) * 100:.2f}%")
    print(f"\t\t\tMCC score: {matthews_corrcoef(label, prediction) * 100:.2f}%")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n")
