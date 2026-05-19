import json
import random

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

NOMES_CLASSES = [
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


def configurar_execucao(semente):

    random.seed(semente)

    np.random.seed(semente)

    tf.keras.utils.set_random_seed(semente)


def carregar_dados(
    diretorio_dados,
    tamanho_imagem,
    tamanho_lote,
    semente
):

    treino = tf.keras.utils.image_dataset_from_directory(
        diretorio_dados,
        validation_split=0.2,
        subset="training",
        seed=semente,
        image_size=(
            tamanho_imagem,
            tamanho_imagem
        ),
        batch_size=tamanho_lote
    )

    validacao = tf.keras.utils.image_dataset_from_directory(
        diretorio_dados,
        validation_split=0.2,
        subset="validation",
        seed=semente,
        image_size=(
            tamanho_imagem,
            tamanho_imagem
        ),
        batch_size=tamanho_lote
    )

    AUTOTUNE = tf.data.AUTOTUNE

    return (
        treino.prefetch(AUTOTUNE),
        validacao.prefetch(AUTOTUNE)
    )


def salvar_grafico_historico(
    historico,
    nome_modelo,
    diretorio_saida
):

    hist = pd.DataFrame(historico.history)

    hist.to_csv(
        diretorio_saida /
        f"historico_{nome_modelo}.csv"
    )

    epocas = range(1, len(hist) + 1)

    fig, eixos = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # Acurácia
    eixos[0].plot(
        epocas,
        hist["accuracy"],
        marker="o"
    )

    eixos[0].plot(
        epocas,
        hist["val_accuracy"],
        marker="o"
    )

    eixos[0].set_title("Acurácia")

    eixos[0].set_xlabel("Épocas")

    eixos[0].set_ylabel("Acurácia")

    eixos[0].legend([
        "Treino",
        "Validação"
    ])

    eixos[0].grid(True)

    # Loss
    eixos[1].plot(
        epocas,
        hist["loss"],
        marker="o"
    )

    eixos[1].plot(
        epocas,
        hist["val_loss"],
        marker="o"
    )

    eixos[1].set_title("Loss")

    eixos[1].set_xlabel("Épocas")

    eixos[1].set_ylabel("Loss")

    eixos[1].legend([
        "Treino",
        "Validação"
    ])

    eixos[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        diretorio_saida /
        f"grafico_{nome_modelo}.png",
        dpi=200
    )

    plt.close()


def salvar_matriz_confusao(
    y_real,
    y_predito,
    nome_modelo,
    diretorio_saida
):

    matriz = confusion_matrix(
        y_real,
        y_predito
    )

    pd.DataFrame(matriz).to_csv(
        diretorio_saida /
        f"matriz_confusao_{nome_modelo}.csv"
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    imagem = ax.imshow(matriz)

    plt.colorbar(imagem)

    ax.set_xticks(
        np.arange(len(NOMES_CLASSES))
    )

    ax.set_yticks(
        np.arange(len(NOMES_CLASSES))
    )

    ax.set_xticklabels(
        NOMES_CLASSES,
        rotation=45
    )

    ax.set_yticklabels(
        NOMES_CLASSES
    )

    ax.set_xlabel(
        "Classe Predita"
    )

    ax.set_ylabel(
        "Classe Real"
    )

    for i in range(len(NOMES_CLASSES)):
        for j in range(len(NOMES_CLASSES)):

            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        diretorio_saida /
        f"matriz_confusao_{nome_modelo}.png",
        dpi=200
    )

    plt.close()


def salvar_predicoes_exemplo(
    imagens,
    labels,
    preds,
    nome_modelo,
    diretorio_saida
):

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(12, 12)
    )

    for ax, img, real, pred in zip(
        axes.ravel(),
        imagens[:25],
        labels[:25],
        preds[:25]
    ):

        ax.imshow(
            img.astype("uint8")
        )

        ax.set_title(
            f"R:{NOMES_CLASSES[real]}\n"
            f"P:{NOMES_CLASSES[pred]}",
            fontsize=8
        )

        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        diretorio_saida /
        f"predicoes_{nome_modelo}.png"
    )

    plt.close()


def avaliar_modelo(
    modelo,
    dataset,
    nome_modelo,
    diretorio_saida
):

    y_real = []

    y_predito = []

    imagens_exemplo = []

    for imagens, labels in dataset:

        preds = modelo.predict(
            imagens,
            verbose=0
        )

        preds = np.argmax(
            preds,
            axis=1
        )

        y_real.extend(
            labels.numpy()
        )

        y_predito.extend(preds)

        imagens_exemplo.extend(
            imagens.numpy()
        )

    salvar_matriz_confusao(
        y_real,
        y_predito,
        nome_modelo,
        diretorio_saida
    )

    salvar_predicoes_exemplo(
        imagens_exemplo,
        y_real,
        y_predito,
        nome_modelo,
        diretorio_saida
    )

    relatorio = classification_report(
        y_real,
        y_predito,
        target_names=NOMES_CLASSES,
        output_dict=True
    )

    pd.DataFrame(relatorio).transpose().to_csv(
        diretorio_saida /
        f"relatorio_classes_{nome_modelo}.csv"
    )

    return {
        "modelo": nome_modelo,

        "acuracia": accuracy_score(
            y_real,
            y_predito
        ),

        "precisao": precision_score(
            y_real,
            y_predito,
            average="weighted"
        ),

        "revocacao": recall_score(
            y_real,
            y_predito,
            average="weighted"
        ),

        "f1_score": f1_score(
            y_real,
            y_predito,
            average="weighted"
        )
    }


def salvar_configuracoes(
    args,
    diretorio_saida
):

    with open(
        diretorio_saida /
        "configuracoes.json",
        "w"
    ) as arquivo:

        json.dump(
            vars(args),
            arquivo,
            indent=4,
            default=str
        )