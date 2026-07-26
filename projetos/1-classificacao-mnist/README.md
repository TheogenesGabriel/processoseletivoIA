# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Theógenes Gabriel Araújo de Andrade

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo implementado em `train_model.py` é uma **Rede Neural Convolucional (CNN)** sequencial composta por **três blocos convolucionais**, seguidos de uma camada densa de classificação:

- **Bloco 1:** `Conv2D` (32 filtros) → `BatchNormalization` → `MaxPooling2D`
- **Bloco 2:** `Conv2D` (64 filtros) → `BatchNormalization` → `MaxPooling2D`
- **Bloco 3:** `Conv2D` (128 filtros) → `BatchNormalization` → `MaxPooling2D`
- **Cabeça de classificação:** `Flatten` → `Dropout` → `Dense` (10 unidades, saída softmax para as 10 classes de dígitos)

O modelo possui um total de **105.098 parâmetros** (104.650 treináveis e 448 não-treináveis, referentes às camadas de Batch Normalization).

A **Batch Normalization** é aplicada após cada camada convolucional, estabilizando e acelerando o treinamento ao normalizar as ativações intermediárias. Já a camada de **Dropout**, posicionada antes da camada densa final, atua como regularização, reduzindo o risco de overfitting ao "desligar" aleatoriamente neurônios durante o treinamento.

Quanto à estratégia de validação, o modelo foi configurado para treinar por até **15 épocas**, com uma divisão de dados de treino/validação. O treinamento foi interrompido automaticamente na **7ª época**, evidenciando o uso de **Early Stopping** monitorando a métrica de validação — mecanismo que interrompe o treinamento quando não há mais ganho relevante de performance, evitando overfitting e economizando tempo de computação.

### 2️⃣ Bibliotecas Utilizadas

O `requirements.txt` do projeto define:

- **TensorFlow** (`>=2.12`) — na execução local, a versão instalada foi a **2.21.0**, trazendo **Keras 3.15.0** embutido, usados para construção, treinamento (`tf.keras`) e conversão/otimização do modelo (`tf.lite`)
- **NumPy** — versão instalada: **2.4.4**, usada na manipulação dos arrays de imagem em `run_inference.py`

### 3️⃣ Técnica de Otimização do Modelo

A técnica utilizada em `optimize_model.py` foi a **Quantização Dinâmica de Faixa (Dynamic Range Quantization)**, aplicada via TensorFlow Lite Converter:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Essa técnica converte os pesos do modelo (originalmente em ponto flutuante de 32 bits) para uma representação de **8 bits inteiros**, mantendo as ativações em ponto flutuante durante a inferência. O resultado é um modelo significativamente menor em disco e mais rápido para execução em dispositivos com recursos limitados (Edge AI), com impacto mínimo na acurácia — trade-off ideal para sistemas embarcados.

### 4️⃣ Resultados Obtidos

- **Acurácia de validação final:** 98,85% (0.9885)
- **Acurácia no conjunto de teste:** 99,08% (0.9908)
- **Tamanho do `model.h5`:** 1.294,88 KB (≈ 1,26 MB)
- **Tamanho do `model.tflite`:** 113,96 KB
- **Redução de tamanho:** 91,2%

### 5️⃣ Comentários Adicionais (Opcional)

Durante a execução do desafio, a principal dificuldade encontrada foi de natureza operacional, relacionada ao ambiente Windows/Git Bash: o ambiente virtual (`.venv`) não estava ativado por padrão em novas sessões do terminal, o que gerava erros de `ModuleNotFoundError` ao tentar rodar os scripts sem ativação prévia (`source .venv/Scripts/activate`).

Também foi identificada uma inversão de conteúdo entre os arquivos `optimize_model.py` e `run_inference.py`: o arquivo `optimize_model.py` continha, por engano, o código de inferência (carregamento direto do `model.tflite`, ainda inexistente naquele momento), o que gerava um erro de `ValueError: Could not open model.tflite`. Após identificar e corrigir a troca — restaurando o código de conversão (`TFLiteConverter` com `Dynamic Range Quantization`) em `optimize_model.py` — o pipeline foi executado com sucesso, do treinamento até a inferência.

Como aprendizado técnico, destaca-se o ganho expressivo obtido com a quantização dinâmica: uma redução de aproximadamente 91% no tamanho do modelo, com impacto mínimo esperado na acurácia, reforçando a relevância dessa técnica para cenários de Edge AI, onde recursos de armazenamento e processamento são limitados.

### 6️⃣ Exemplo de Inferência

```
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

O modelo otimizado (`model.tflite`) acertou **100% das 5 amostras testadas** do conjunto de teste do MNIST, incluindo dígitos com formas visualmente distintas (0, 1, 2, 4, 7). O resultado confirma que, apesar da redução de ~91% no tamanho do arquivo por meio da quantização dinâmica, o modelo manteve sua capacidade de classificação intacta nesse conjunto de amostras — evidência de que a técnica de otimização não comprometeu a qualidade das predições.