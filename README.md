# Classificação de flores Iris

Projeto de estudo de machine learning que classifica flores Iris em três espécies (setosa, versicolor e virginica) a partir de quatro medidas das pétalas e sépalas. O notebook percorre todas as etapas do problema: análise exploratória, uma regra de classificação feita à mão, regressão logística, validação cruzada, ajuste de hiperparâmetro e avaliação final no conjunto de teste.

## Dataset

O dataset Iris (Fisher, 1936) vem junto com o scikit-learn (`sklearn.datasets.load_iris()`), então não é preciso baixar nada.

- 150 flores, 50 de cada espécie
- 4 medidas em centímetros: comprimento e largura da sépala, comprimento e largura da pétala
- Alvo: a espécie da flor (0 = setosa, 1 = versicolor, 2 = virginica)

## Por que regressão logística?

A escolha do modelo segue do tipo de problema:

1. **Os dados têm rótulos.** Cada flor vem com a espécie correta (`target`). Quando existe uma resposta conhecida para o modelo aprender, o problema é de **aprendizado supervisionado**: o modelo aprende, a partir de exemplos, a ligar as medidas à espécie. Sem rótulos, seria aprendizado não supervisionado, como agrupar as flores com k-means sem saber as espécies.
2. **O rótulo é uma categoria.** A resposta é "qual espécie", não um número contínuo. Então o problema é de **classificação**, e não de regressão. Como são três espécies, é classificação **multiclasse**.
3. **As classes são quase separáveis por fronteiras simples.** A análise exploratória mostra que a setosa fica totalmente separada das outras e que versicolor e virginica se sobrepõem pouco. Um modelo com fronteiras lineares dá conta disso.

Com esse cenário, a regressão logística é um bom ponto de partida:

- **É um classificador, apesar do nome.** Ela calcula uma soma ponderada das 4 medidas para cada espécie e transforma essas pontuações em probabilidades (função softmax). A previsão é a espécie mais provável.
- **É simples e rápida.** Funciona bem como primeiro modelo, antes de testar algo mais complexo.
- **É interpretável.** Cada espécie tem um peso para cada medida (`model.coef_`), então dá para ver quais medidas pesam mais na decisão.
- **Combina com poucos dados.** Com 150 flores e 4 medidas, um modelo simples e regularizado (pelo parâmetro `C`) corre menos risco de overfitting do que modelos muito flexíveis.
- **Dá probabilidades, não só a classe.** Com `predict_proba`, dá para medir a confiança de cada previsão.

Como as classes são balanceadas (50 flores de cada), a **acurácia** é uma métrica adequada. Com classes desbalanceadas, seria melhor olhar também precisão, recall e F1.

Outros modelos também servem para esse tipo de problema, como KNN, árvore de decisão, SVM e Naive Bayes. Compará-los está nos próximos passos.

## Etapas do notebook

1. **Análise exploratória:** estatísticas descritivas, histogramas e gráfico em pares. As medidas de pétala separam bem as espécies, mas versicolor e virginica se sobrepõem um pouco.
2. **Divisão treino/teste:** 80/20, estratificada e com `random_state=42`. São 120 flores para treino e 30 para teste. O teste só é usado no final.
3. **Baseline:** chutando uma espécie ao acaso, a acurácia esperada é de 33,3%.
4. **Regra manual:** classifica só pelo comprimento da pétala (menos de 2,5 cm → setosa; menos de 4,8 cm → versicolor; o resto → virginica).
5. **Regressão logística:** primeiro com uma divisão de validação (holdout) e depois com validação cruzada de 5 folds no conjunto de treino.
6. **Análise dos erros:** com `cross_val_predict`, cada flor do treino recebe uma previsão feita por um modelo que não a viu. Os erros aparecem num gráfico de comprimento × largura da pétala.
7. **Ajuste de hiperparâmetro:** valores de `C` (o inverso da força da regularização) comparados na validação cruzada.
8. **Avaliação final:** o modelo escolhido é treinado com todo o treino e avaliado uma única vez no teste.

## Resultados

Divisão estratificada com `random_state=42`:

| Modelo | Treino | Validação cruzada (5 folds) | Teste (30 flores) |
|---|---|---|---|
| Regra manual (comprimento da pétala) | 95,0% | — | — |
| Regressão logística (`C=2`, `max_iter=200`) | 98,3%¹ | 97,5% | 96,7% |

¹ `model.score(x_train, y_train)` depois do treino final com as 120 flores.

- Treino, validação cruzada e teste ficam próximos, então não há sinal de overfitting.
- O único erro no teste é uma versicolor com pétala de 5,0 × 1,7 cm, prevista como virginica. Ela fica justamente na faixa em que as duas espécies se misturam.
- Os dois modelos passam com folga dos 33,3% do chute. A regra manual, que usa uma única medida, já chega a 95% no treino: o comprimento da pétala sozinho separa bem as espécies.

## O que aprendi

- **Acertar 100% no teste não é overfitting.** Overfitting é quando o modelo vai muito melhor no treino do que em dados novos. Com um teste de 30 flores, cada erro vale 3,3 pontos percentuais. Em 1000 divisões aleatórias 80/20 simuladas, o teste deu 100% em cerca de 30% das vezes.
- **Uma única divisão engana.** O resultado muda conforme quais flores caem no treino e no teste. A validação cruzada dá uma estimativa mais estável. Nas simulações, 80/20 e 75/25 deram a mesma média (~96%).
- **Reprodutibilidade:** `random_state` fixa o sorteio e `stratify` mantém a proporção das espécies em cada parte. Sem eles, cada execução dá números diferentes.
- **Hiperparâmetros se ajustam na validação cruzada, nunca no teste.** E vale testar em escala logarítmica (0,01; 0,1; 1; 10; 100): numa faixa estreita, como de 1 a 5, os resultados tendem a empatar.
- **Rodar o notebook do início ao fim** (Kernel → Restart Kernel and Run All Cells) antes de tirar conclusões, para garantir que as saídas correspondem ao código.

## Como executar

Com Python 3.10 ou mais recente:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook IrisDataset.ipynb
```

## Estrutura

```
.
├── IrisDataset.ipynb   # notebook com todo o projeto
└── README.md
```

## Tecnologias

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn e Jupyter.

## Próximos passos

- Fazer o ajuste de `C` com `GridSearchCV`.
- Colocar `StandardScaler` e o modelo num `Pipeline`.
- Comparar com outros modelos, como KNN, árvore de decisão e SVM.
- Analisar a matriz de confusão e o `classification_report`.
- Usar validação cruzada repetida para medir a variação entre divisões.

