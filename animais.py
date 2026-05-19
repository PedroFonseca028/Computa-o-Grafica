import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import random

CLASSES = [
    "Cachorro", "Cavalo", "Elefante", "Borboleta", "Galinha",
    "Gato", "Vaca", "Ovelha", "Aranha", "Esquilo"
]


class DatasetManager:
    def __init__(self, root, image_size, batch, seed):
        self.root = root
        self.image_size = image_size
        self.batch = batch
        self.seed = seed

    def load(self):
        train = tf.keras.utils.image_dataset_from_directory(
            self.root,
            validation_split=0.2,
            subset="training",
            seed=self.seed,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch
        )

        valid = tf.keras.utils.image_dataset_from_directory(
            self.root,
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch
        )

        return train.prefetch(tf.data.AUTOTUNE), valid.prefetch(tf.data.AUTOTUNE)


class ModelFactory:
    @staticmethod
    def compact(input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        return ModelFactory.compile(model)

    @staticmethod
    def regularized(input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        return ModelFactory.compile(model)

    @staticmethod
    def compile(model):
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model


class Reporter:
    def __init__(self, output):
        self.output = output

    def save_training_curves(self, history, label):
        df = pd.DataFrame(history.history)
        df.to_csv(self.output / f"{label}_training.csv")

        df[['accuracy', 'val_accuracy']].plot()
        plt.savefig(self.output / f"{label}_acc.png")
        plt.close()

        df[['loss', 'val_loss']].plot()
        plt.savefig(self.output / f"{label}_loss.png")
        plt.close()

    def save_evaluation(self, y_true, y_pred, label):
        cm = confusion_matrix(y_true, y_pred)

        pd.DataFrame(cm).to_csv(self.output / f"{label}_cm.csv")

        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASSES,
            output_dict=True
        )

        pd.DataFrame(report).transpose().to_csv(
            self.output / f"{label}_report.csv"
        )


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.output.mkdir(exist_ok=True)

    def predict(self, model, ds):
        y_true, y_pred = [], []

        for x, y in ds:
            pred = np.argmax(model.predict(x, verbose=0), axis=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred)

        return y_true, y_pred

    def run(self):
        manager = DatasetManager(
            self.args.data_dir,
            self.args.img_size,
            self.args.batch_size,
            self.args.seed
        )

        train, valid = manager.load()
        reporter = Reporter(self.output)

        architectures = {
            "compact": ModelFactory.compact,
            "regularized": ModelFactory.regularized
        }

        summary = []

        for name, builder in architectures.items():
            print(f"Executando {name}")

            net = builder((self.args.img_size, self.args.img_size, 3))

            hist = net.fit(
                train,
                validation_data=valid,
                epochs=self.args.epochs
            )

            reporter.save_training_curves(hist, name)

            y_true, y_pred = self.predict(net, valid)
            reporter.save_evaluation(y_true, y_pred, name)

            acc = np.mean(np.array(y_true) == np.array(y_pred))

            summary.append({
                "modelo": name,
                "accuracy": acc
            })

        pd.DataFrame(summary).to_csv(
            self.output / "summary.csv",
            index=False
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw-img"))
    parser.add_argument("--output-dir", type=Path, default=Path("saida"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    seed_everything(args.seed)

    with open(args.output_dir / "run.json", "w") as f:
        json.dump(vars(args), f, default=str)

    Experiment(args).run()