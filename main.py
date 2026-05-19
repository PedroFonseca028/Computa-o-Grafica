import argparse
from pathlib import Path

import pandas as pd

from modelos import (
    criar_cnn_dropout,
    criar_cnn_pequena
)

from utils import (
    avaliar_modelo,
    carregar_dados,
    configurar_execucao,
    salvar_configuracoes,
    salvar_grafico_historico
)


def argumentos():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epocas", type=int, default=8)

    parser.add_argument(
        "--tamanho-lote",
        type=int,
        default=16
    )

    parser.add_argument(
        "--tamanho-imagem",
        type=int,
        default=64
    )

    parser.add_argument(
        "--diretorio-dados",
        type=Path,
        default=Path("data/raw-img")
    )

    parser.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("resultados")
    )

    parser.add_argument(
        "--semente",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():

    args = argumentos()

    args.diretorio_saida.mkdir(exist_ok=True)

    configurar_execucao(args.semente)

    salvar_configuracoes(
        args,
        args.diretorio_saida
    )

    treino, validacao = carregar_dados(
        args.diretorio_dados,
        args.tamanho_imagem,
        args.tamanho_lote,
        args.semente
    )

    modelos = {
        "cnn_pequena": criar_cnn_pequena,
        "cnn_dropout": criar_cnn_dropout
    }

    resultados = []

    for nome, construtor in modelos.items():

        print(f"\nTreinando modelo: {nome}")

        modelo = construtor(
            (
                args.tamanho_imagem,
                args.tamanho_imagem,
                3
            )
        )

        historico = modelo.fit(
            treino,
            validation_data=validacao,
            epochs=args.epocas,
            verbose=1
        )

        salvar_grafico_historico(
            historico,
            nome,
            args.diretorio_saida
        )

        metricas = avaliar_modelo(
            modelo,
            validacao,
            nome,
            args.diretorio_saida
        )

        resultados.append(metricas)

    pd.DataFrame(resultados).to_csv(
        args.diretorio_saida / "resumo_metricas.csv",
        index=False
    )

    print("\nExecução concluída com sucesso")


if __name__ == "__main__":
    main()