import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_NAMES = [
    "Cachorro",
    "Cavalo",
    "Elefante",
    "Borboleta",
    "Galinha",
    "Gato",
    "Vaca",
    "Ovelha",
    "Aranha",
    "Esquilo",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw-img"))
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def configure_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(data_dir, img_size, batch_size, seed):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def build_cnn_pequena(input_shape=(64, 64, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_cnn_dropout(input_shape=(64, 64, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_history_plot(history, model_name, output_dir):
    hist = pd.DataFrame(history.history)
    hist.to_csv(output_dir / f"historico_{model_name}.csv")

    epochs = range(1, len(hist)+1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, hist["accuracy"], marker="o")
    axes[0].plot(epochs, hist["val_accuracy"], marker="o")
    axes[0].set_title("Acurácia")
    axes[0].grid(True)
    axes[0].legend(["Treino", "Validação"])

    axes[1].plot(epochs, hist["loss"], marker="o")
    axes[1].plot(epochs, hist["val_loss"], marker="o")
    axes[1].set_title("Loss")
    axes[1].grid(True)
    axes[1].legend(["Treino", "Validação"])

    fig.tight_layout()
    fig.savefig(output_dir / f"convergencia_{model_name}.png", dpi=200)
    plt.close()


def save_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(cm).to_csv(
        output_dir / f"matriz_confusao_{model_name}.csv"
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm)
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45)
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / f"matriz_confusao_{model_name}.png", dpi=200)
    plt.close()


def save_sample_predictions(images, labels, preds, model_name, output_dir):
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for ax, img, real, pred in zip(
        axes.ravel(),
        images[:25],
        labels[:25],
        preds[:25]
    ):
        ax.imshow(img.astype("uint8"))
        ax.set_title(
            f"R:{CLASS_NAMES[real]}\nP:{CLASS_NAMES[pred]}",
            fontsize=8
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"predicoes_{model_name}.png")
    plt.close()


def evaluate_model(model, dataset, model_name, output_dir):
    y_true = []
    y_pred = []
    sample_images = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        sample_images.extend(images.numpy())

    save_confusion_matrix(y_true, y_pred, model_name, output_dir)
    save_sample_predictions(sample_images, y_true, y_pred, model_name, output_dir)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"relatorio_classes_{model_name}.csv"
    )

    return {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }


def save_config(args, output_dir):
    with open(output_dir / "config_execucao.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)


def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)

    configure_runtime(args.seed)
    save_config(args, args.output_dir)

    train_ds, val_ds = load_data(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.seed
    )

    modelos = {
        "cnn_pequena": build_cnn_pequena,
        "cnn_dropout": build_cnn_dropout
    }

    resultados = []

    for nome, builder in modelos.items():
        print(f"\nTreinando {nome}")

        model = builder((args.img_size, args.img_size, 3))

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            verbose=1
        )

        save_history_plot(history, nome, args.output_dir)

        metricas = evaluate_model(
            model,
            val_ds,
            nome,
            args.output_dir
        )

        resultados.append(metricas)

    pd.DataFrame(resultados).to_csv(
        args.output_dir / "metricas_resumo.csv",
        index=False
    )

    print("\nConcluído")


if __name__ == "__main__":
    main()