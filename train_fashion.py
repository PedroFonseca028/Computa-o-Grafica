#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina CNNs no Fashion-MNIST e salva métricas/gráficos.")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamanho do lote.")
    parser.add_argument(
        "--modelos",
        nargs="+",
        default=["cnn_pequena", "cnn_dropout"],
        choices=["cnn_pequena", "cnn_dropout"],
        help="Modelos a treinar.",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Limita exemplos de treino.")
    parser.add_argument("--test-limit", type=int, default=None, help="Limita exemplos de teste.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraçao do treino para validação.")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade.")
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"), help="Pasta de saída.")
    parser.add_argument("--salvar-modelos", action="store_true", help="Salva os modelos .keras.")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace):
    if args.epochs < 1:
        raise ValueError("epochs deve ser >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size deve ser > 0")
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("train-limit deve ser > 0")
    if args.test_limit is not None and args.test_limit <= 0:
        raise ValueError("test-limit deve ser > 0")
    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("val-split deve estar em [0.0, 1.0)")


def configure_runtime(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def load_data(train_limit: Optional[int] = None, test_limit: Optional[int] = None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    if train_limit is not None:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit is not None:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    return (x_train, y_train), (x_test, y_test)


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_pequena(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_pequena")
    return compile_model(model)


def build_cnn_dropout(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_dropout")
    return compile_model(model)


def make_model(name: str) -> tf.keras.Model:
    builders = {
        "cnn_pequena": build_cnn_pequena,
        "cnn_dropout": build_cnn_dropout,
    }
    return builders[name]()


def split_train_validation(x_train, y_train, val_split: float, seed: int):
    if val_split <= 0:
        return x_train, y_train, None, None
    return train_test_split(x_train, y_train, test_size=val_split, random_state=seed, stratify=y_train)


def save_run_config(args: argparse.Namespace, output_dir: Path):
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "modelos": args.modelos,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "val_split": args.val_split,
        "seed": args.seed,
        "tensorflow_version": tf.__version__,
    }
    with (output_dir / "config_execucao.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_history_plot(history, model_name: str, output_dir: Path):
    hist = pd.DataFrame(history.history)
    hist.to_csv(output_dir / f"historico_{model_name}.csv", index_label="epoca")
    epochs = np.arange(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist["accuracy"], marker="o", label="Treino")
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], marker="o", label="Validacao")
    axes[0].set_title("Acuracia")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Acuracia")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, hist["loss"], marker="o", label="Treino")
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"], marker="o", label="Validacao")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Convergencia - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"convergencia_{model_name}.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, model_name: str, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / f"matriz_confusao_{model_name}.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(f"Matriz de confusao - {model_name}")
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"matriz_confusao_{model_name}.png", dpi=150)
    plt.close(fig)


def save_sample_predictions(x_test, y_test, y_pred, model_name: str, output_dir: Path, n: int = 25):
    n = min(n, len(x_test))
    if n == 0:
        return
    rows = cols = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).ravel()

    for i in range(n):
        ax = axes[i]
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        true_label = int(y_test[i])
        pred_label = int(y_pred[i])
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"{CLASS_NAMES[true_label]}\n-> {CLASS_NAMES[pred_label]}", color=color, fontsize=8)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Amostras de predicao - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"predicoes_{model_name}.png", dpi=150)
    plt.close(fig)


def evaluate_model(model: tf.keras.Model, x_test, y_test, model_name: str, output_dir: Path) -> Dict:
    if len(x_test) == 0:
        raise ValueError("x_test está vazio; não é possível avaliar.")

    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / f"relatorio_classes_{model_name}.csv")

    save_confusion_matrix_plot(y_test, y_pred, model_name, output_dir)
    save_sample_predictions(x_test, y_test, y_pred, model_name, output_dir)

    return {
        "modelo": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "parametros": int(model.count_params()),
    }


def run_training(args: argparse.Namespace):
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    configure_runtime(args.seed)
    save_run_config(args, outdir)
    (x_train, y_train), (x_test, y_test) = load_data(args.train_limit, args.test_limit)
    x_train_fit, x_val, y_train_fit, y_val = split_train_validation(x_train, y_train, args.val_split, args.seed)

    summaries = []
    for model_name in args.modelos:
        print(f"\n=== Treinando {model_name} ===")
        model = make_model(model_name)
        model.summary()

        history = model.fit(
            x_train_fit,
            y_train_fit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val) if x_val is not None else None,
            verbose=2,
        )

        save_history_plot(history, model_name, outdir)
        summaries.append(evaluate_model(model, x_test, y_test, model_name, outdir))

        if args.salvar_modelos:
            model.save(outdir / f"{model_name}.keras")

    summary_df = pd.DataFrame(summaries).sort_values("f1_weighted", ascending=False)
    summary_df.to_csv(outdir / "metricas_resumo.csv", index=False)

    print("\nResumo dos resultados no conjunto de teste:")
    print(summary_df.to_string(index=False))
    print(f"\nArquivos salvos em: {outdir.resolve()}")


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()