# Fashion MNIST Autoencoder — Deep Learning

Autoencoder convolucional desenvolvido em **TensorFlow/Keras** para aprender **representações latentes de roupas** do dataset **Fashion MNIST**.

O modelo aprende a:

- comprimir imagens em um **espaço latente de 2 dimensões**
- reconstruir imagens a partir desse espaço
- visualizar **clusters de roupas** no espaço latente

Este projeto demonstra conceitos importantes de **Deep Learning** e **Representation Learning**.

---

# Conceito

Um **Autoencoder** é uma rede neural composta por duas partes.

## Encoder

Transforma a imagem em uma **representação comprimida (embedding)**.

Imagem (32x32)
↓
Conv2D
↓
Conv2D
↓
Conv2D
↓
Flatten
↓
Latent Space (2D)


## Decoder

Reconstrói a imagem original a partir do vetor latente.


Latent Vector (2D)
↓
Dense
↓
Reshape
↓
Conv2DTranspose
↓
Conv2DTranspose
↓
Conv2DTranspose
↓
Imagem Reconstruída (32x32)


⚙️ Treinamento

Configuração utilizada:

optimizer = "adam"
loss = "binary_crossentropy"
epochs = 5
batch_size = 100

Treinamento do modelo:

autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Observação:

O alvo do treinamento é a própria imagem de entrada.

input → encoder → latent space → decoder → reconstrução

# Modelos Salvos

Após o treinamento são salvos três modelos:

models/
├── autoencoder.keras
├── encoder.keras
└── decoder.keras

Isso permite utilizar separadamente:

Encoder → gerar embeddings

Decoder → gerar novas imagens

Autoencoder → reconstrução completa

# Reconstrução de Imagens

Após o treinamento o modelo reconstrói imagens do dataset.

predictions = autoencoder.predict(example_images)

Fluxo de processamento:

Imagem → Encoder → Latent Space → Decoder → Imagem Reconstruída

# Espaço Latente (2D)

Como o espaço latente possui 2 dimensões, podemos visualizá-lo diretamente.

plt.scatter(embeddings[:,0], embeddings[:,1])

Cada ponto representa uma imagem do dataset.

Clusters indicam que o modelo aprendeu características visuais semelhantes.

Exemplo esperado:

sneakers próximos de sneakers

bolsas próximas de bolsas

casacos próximos de casacos

# Visualização por Classe

Também podemos colorir os embeddings usando os labels:

plt.scatter(
    embeddings[:,0],
    embeddings[:,1],
    c=example_labels,
    cmap="rainbow"
)

Isso permite analisar:

separação entre classes

qualidade da representação aprendida

# Gerando Novas Imagens

Podemos gerar novas roupas amostrando pontos aleatórios no espaço latente.

sample = np.random.uniform(mins, maxs, size=(18,2))
reconstructions = decoder.predict(sample)

# 🚀 Como Executar
1️⃣ Clonar o repositório
git clone https://github.com/seu-usuario/fashion-mnist-autoencoder.git
2️⃣ Instalar dependências
pip install tensorflow matplotlib numpy
3️⃣ Executar o projeto
python main.py
🧩 Tecnologias Utilizadas

Python

TensorFlow

Keras

NumPy

Matplotlib

# 🎯 Aplicações de Autoencoders

Autoencoders podem ser usados em diversas tarefas de Machine Learning:

redução de dimensionalidade

compressão de imagens

detecção de anomalias

geração de dados

pré-treinamento de redes neurais

# 👩‍💻 Autora

Mayara Guerra Cabral

Técnica em Inteligência Artificial
Instituto Metrópole Digital — UFRN
