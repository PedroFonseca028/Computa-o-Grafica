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

CLASSES_ANIMAIS = [
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


# ==========================================
# CONFIGURAÇÕES
# ==========================================

def obter_argumentos():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epocas", type=int, default=8)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/raw-img")
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("saida_modelos")
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def definir_semente(seed):

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)


# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

def preparar_datasets(
    dataset_path,
    image_size,
    batch_size,
    seed
):

    treino_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    validacao_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size
    )

    autotune = tf.data.AUTOTUNE

    treino_dataset = treino_dataset.prefetch(autotune)

    validacao_dataset = validacao_dataset.prefetch(autotune)

    return treino_dataset, validacao_dataset


# ==========================================
# MODELO CNN PEQUENA
# ==========================================

def montar_rede_basica(
    input_shape=(64, 64, 3),
    total_classes=10
):

    rede = tf.keras.Sequential([

        tf.keras.layers.Rescaling(
            1./255,
            input_shape=input_shape
        ),

        tf.keras.layers.Conv2D(
            16,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            total_classes,
            activation="softmax"
        )
    ])

    rede.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return rede


# ==========================================
# MODELO CNN PROFUNDA
# ==========================================

def montar_rede_profunda(
    input_shape=(64, 64, 3),
    total_classes=10
):

    rede = tf.keras.Sequential([

        tf.keras.layers.Rescaling(
            1./255,
            input_shape=input_shape
        ),

        tf.keras.layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            3,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            total_classes,
            activation="softmax"
        )
    ])

    rede.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return rede


# ==========================================
# GRÁFICOS
# ==========================================

def gerar_grafico_treinamento(
    historico,
    nome_rede,
    pasta_saida
):

    dataframe_historico = pd.DataFrame(
        historico.history
    )

    dataframe_historico.to_csv(
        pasta_saida / f"dados_treinamento_{nome_rede}.csv"
    )

    lista_epocas = range(
        1,
        len(dataframe_historico) + 1
    )

    figura, eixo = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # Acurácia
    eixo[0].plot(
        lista_epocas,
        dataframe_historico["accuracy"],
        marker="o"
    )

    eixo[0].plot(
        lista_epocas,
        dataframe_historico["val_accuracy"],
        marker="o"
    )

    eixo[0].set_title("Acurácia")

    eixo[0].set_xlabel("Épocas")

    eixo[0].set_ylabel("Valor")

    eixo[0].legend([
        "Treino",
        "Validação"
    ])

    eixo[0].grid(True)

    # Loss
    eixo[1].plot(
        lista_epocas,
        dataframe_historico["loss"],
        marker="o"
    )

    eixo[1].plot(
        lista_epocas,
        dataframe_historico["val_loss"],
        marker="o"
    )

    eixo[1].set_title("Loss")

    eixo[1].set_xlabel("Épocas")

    eixo[1].set_ylabel("Valor")

    eixo[1].legend([
        "Treino",
        "Validação"
    ])

    eixo[1].grid(True)

    figura.tight_layout()

    figura.savefig(
        pasta_saida / f"grafico_treinamento_{nome_rede}.png",
        dpi=200
    )

    plt.close()


# ==========================================
# MATRIZ DE CONFUSÃO
# ==========================================

def gerar_matriz_confusao(
    y_real,
    y_predito,
    nome_rede,
    pasta_saida
):

    matriz = confusion_matrix(
        y_real,
        y_predito
    )

    pd.DataFrame(matriz).to_csv(
        pasta_saida / f"valores_matriz_{nome_rede}.csv"
    )

    figura, eixo = plt.subplots(
        figsize=(12, 10)
    )

    imagem = eixo.imshow(matriz)

    plt.colorbar(imagem)

    eixo.set_xticks(
        np.arange(len(CLASSES_ANIMAIS))
    )

    eixo.set_yticks(
        np.arange(len(CLASSES_ANIMAIS))
    )

    eixo.set_xticklabels(
        CLASSES_ANIMAIS,
        rotation=45
    )

    eixo.set_yticklabels(
        CLASSES_ANIMAIS
    )

    eixo.set_xlabel("Classe Predita")

    eixo.set_ylabel("Classe Real")

    for i in range(len(CLASSES_ANIMAIS)):
        for j in range(len(CLASSES_ANIMAIS)):

            eixo.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center"
            )

    figura.tight_layout()

    figura.savefig(
        pasta_saida / f"imagem_matriz_{nome_rede}.png",
        dpi=200
    )

    plt.close()


# ==========================================
# EXEMPLOS DE PREDIÇÃO
# ==========================================

def gerar_imagens_predicao(
    imagens,
    labels,
    predicoes,
    nome_rede,
    pasta_saida
):

    figura, eixos = plt.subplots(
        5,
        5,
        figsize=(12, 12)
    )

    for eixo, imagem, real, previsto in zip(
        eixos.ravel(),
        imagens[:25],
        labels[:25],
        predicoes[:25]
    ):

        eixo.imshow(
            imagem.astype("uint8")
        )

        eixo.set_title(
            f"R: {CLASSES_ANIMAIS[real]}\nP: {CLASSES_ANIMAIS[previsto]}",
            fontsize=8
        )

        eixo.axis("off")

    figura.tight_layout()

    figura.savefig(
        pasta_saida / f"predicoes_visuais_{nome_rede}.png"
    )

    plt.close()


# ==========================================
# AVALIAÇÃO
# ==========================================

def analisar_modelo(
    modelo,
    dataset,
    nome_rede,
    pasta_saida
):

    classes_reais = []

    classes_previstas = []

    imagens_salvas = []

    for imagens, labels in dataset:

        probabilidades = modelo.predict(
            imagens,
            verbose=0
        )

        previsoes = np.argmax(
            probabilidades,
            axis=1
        )

        classes_reais.extend(
            labels.numpy()
        )

        classes_previstas.extend(
            previsoes
        )

        imagens_salvas.extend(
            imagens.numpy()
        )

    gerar_matriz_confusao(
        classes_reais,
        classes_previstas,
        nome_rede,
        pasta_saida
    )

    gerar_imagens_predicao(
        imagens_salvas,
        classes_reais,
        classes_previstas,
        nome_rede,
        pasta_saida
    )

    relatorio = classification_report(
        classes_reais,
        classes_previstas,
        target_names=CLASSES_ANIMAIS,
        output_dict=True
    )

    pd.DataFrame(relatorio).transpose().to_csv(
        pasta_saida / f"metricas_classes_{nome_rede}.csv"
    )

    return {

        "modelo": nome_rede,

        "accuracy": accuracy_score(
            classes_reais,
            classes_previstas
        ),

        "precision": precision_score(
            classes_reais,
            classes_previstas,
            average="weighted"
        ),

        "recall": recall_score(
            classes_reais,
            classes_previstas,
            average="weighted"
        ),

        "f1_score": f1_score(
            classes_reais,
            classes_previstas,
            average="weighted"
        )
    }


# ==========================================
# CONFIGURAÇÕES EXECUÇÃO
# ==========================================

def salvar_parametros_execucao(
    argumentos,
    pasta_saida
):

    with open(
        pasta_saida / "parametros_execucao.json",
        "w"
    ) as arquivo:

        json.dump(
            vars(argumentos),
            arquivo,
            indent=4,
            default=str
        )


# ==========================================
# MAIN
# ==========================================

def main():

    args = obter_argumentos()

    args.output_path.mkdir(
        exist_ok=True
    )

    definir_semente(args.seed)

    salvar_parametros_execucao(
        args,
        args.output_path
    )

    treino_dataset, validacao_dataset = preparar_datasets(
        args.dataset_path,
        args.image_size,
        args.batch_size,
        args.seed
    )

    arquiteturas = {

        "rede_basica": montar_rede_basica,

        "rede_profunda": montar_rede_profunda
    }

    lista_resultados = []

    parada_antecipada = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True
    )

    for nome_modelo, funcao_modelo in arquiteturas.items():

        print(f"\nTreinando arquitetura: {nome_modelo}")

        rede = funcao_modelo(

            (args.image_size, args.image_size, 3)
        )

        historico = rede.fit(

            treino_dataset,

            validation_data=validacao_dataset,

            epochs=args.epocas,

            callbacks=[parada_antecipada],

            verbose=1
        )

        gerar_grafico_treinamento(
            historico,
            nome_modelo,
            args.output_path
        )

        metricas = analisar_modelo(
            rede,
            validacao_dataset,
            nome_modelo,
            args.output_path
        )

        lista_resultados.append(
            metricas
        )

    pd.DataFrame(lista_resultados).to_csv(
        args.output_path / "resultado_final_modelos.csv",
        index=False
    )

    print("\nTreinamento finalizado com sucesso")


if __name__ == "__main__":
    main()