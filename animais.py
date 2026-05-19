import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


CLASSES = [
    "Cachorro",
    "Cavalo",
    "Elefante",
    "Borboleta",
    "Galinha",
    "Gato",
    "Vaca",
    "Ovelha",
    "Aranha",
    "Esquilo"
]


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class DataLoader:

    def __init__(self, path, img_size, batch_size, seed):
        self.path = path
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

    def load(self):

        train = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="training",
            seed=self.seed,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )

        val = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )

        return (
            train.prefetch(tf.data.AUTOTUNE),
            val.prefetch(tf.data.AUTOTUNE)
        )


class ModelBuilder:

    @staticmethod
    def basic(shape):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.Rescaling(1./255),

            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        return ModelBuilder.compile(model)

    @staticmethod
    def dropout(shape):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.Rescaling(1./255),

            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(10, activation="softmax")
        ])

        return ModelBuilder.compile(model)

    @staticmethod
    def compile(model):

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


class SaveResults:

    def __init__(self, output):
        self.output = output

    def history(self, history, name):

        df = pd.DataFrame(history.history)
        df.to_csv(self.output / f"{name}_history.csv")

        fig, ax = plt.subplots()
        ax.plot(df["accuracy"])
        ax.plot(df["val_accuracy"])
        ax.legend(["Treino", "Validação"])
        ax.set_title("Acurácia")
        plt.savefig(self.output / f"{name}_accuracy.png")
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(df["loss"])
        ax.plot(df["val_loss"])
        ax.legend(["Treino", "Validação"])
        ax.set_title("Loss")
        plt.savefig(self.output / f"{name}_loss.png")
        plt.close()

    def confusion(self, real, pred, name):

        cm = confusion_matrix(real, pred)

        pd.DataFrame(cm).to_csv(
            self.output / f"{name}_confusion.csv"
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(cm)

        plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
        plt.yticks(range(len(CLASSES)), CLASSES)

        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                plt.text(j, i, cm[i, j], ha="center")

        plt.tight_layout()
        plt.savefig(self.output / f"{name}_confusion.png")
        plt.close()

    def samples(self, images, real, pred, name):

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))

        for ax, img, r, p in zip(
            axes.ravel(),
            images[:25],
            real[:25],
            pred[:25]
        ):
            ax.imshow(img.astype("uint8"))
            ax.set_title(
                f"{CLASSES[r]} → {CLASSES[p]}",
                fontsize=8
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output / f"{name}_samples.png")
        plt.close()

    def report(self, real, pred, name):

        report = classification_report(
            real,
            pred,
            target_names=CLASSES,
            output_dict=True
        )

        pd.DataFrame(report).transpose().to_csv(
            self.output / f"{name}_report.csv"
        )


class Experiment:

    def __init__(self, args):
        self.args = args
        self.writer = SaveResults(args.output_dir)

    def predict(self, model, dataset):

        imgs, real, pred = [], [], []

        for x, y in dataset:
            p = np.argmax(model.predict(x, verbose=0), axis=1)

            imgs.extend(x.numpy())
            real.extend(y.numpy())
            pred.extend(p)

        return imgs, real, pred

    def run(self):

        train, val = DataLoader(
            self.args.data_dir,
            self.args.img_size,
            self.args.batch_size,
            self.args.seed
        ).load()

        models = {
            "baseline": ModelBuilder.basic,
            "dropout": ModelBuilder.dropout
        }

        summary = []

        for name, builder in models.items():

            print(f"\nTreinando {name}")

            model = builder(
                (self.args.img_size, self.args.img_size, 3)
            )

            start = time.time()

            history = model.fit(
                train,
                validation_data=val,
                epochs=self.args.epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=2,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )

            duration = time.time() - start

            self.writer.history(history, name)

            imgs, y_true, y_pred = self.predict(model, val)

            self.writer.confusion(y_true, y_pred, name)
            self.writer.samples(imgs, y_true, y_pred, name)
            self.writer.report(y_true, y_pred, name)

            summary.append({
                "modelo": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted"),
                "tempo_segundos": duration
            })

        pd.DataFrame(summary).to_csv(
            self.args.output_dir / "metricas.csv",
            index=False
        )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=64)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw-img")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saida")
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    args.output_dir.mkdir(exist_ok=True)

    configure_seed(args.seed)

    with open(args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    Experiment(args).run()

    print("\nConcluído")