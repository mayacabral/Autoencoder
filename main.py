import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models, callbacks
import tensorflow.keras.backend as K
from utils import display


(x_train, y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

plt.figure(figsize=(8,8))
for idx in range(25):
    plt.subplot(5, 5, idx + 1)
    plt.imshow(x_train[idx], cmap="gray")
    plt.title(classes[y_train[idx]], fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()

def preprocess_fashion_mnist(imgs):
    """
    imgs: (N, 28, 28) ou (28, 28)
    retorno: (N, 32, 32, 1) ou (32, 32, 1), float32 em [0, 1]
    """
    # normaliza para [0, 1]
    imgs = imgs.astype(np.float32) / 255.0

    # padding: 28 -> 32 (2 pixels em cada lado)
    imgs = np.pad(imgs, pad_width=((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0.0)

    # expandindo uma dimensão para usar camadas convolucionais
    imgs = np.expand_dims(imgs,-1)

    return imgs


x_train = preprocess_fashion_mnist(x_train)
x_test = preprocess_fashion_mnist(x_test)

#Encoder

#1. define a camada de entrada do encoder
encoder_input = layers.Input(shape=(32,32,1),name="encoder_input")

#2. Empilha camadas convolucionais (2d) em sequencia no topo de cada
x = layers.Conv2D(32, (3,3), strides=2, activation='relu', padding='same')(encoder_input)
x = layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3,3), strides=2, activation='relu', padding='same')(x)

shape_before_flattening = K.int_shape(x)[1:]

#3. Transforma a ultima camada convolucional em um vetor (1D)
x = layers.Flatten()(x)

#4. Conecta o vetor a o embedding de 2 dimensões (2D) com a camada
encoder_output = layers.Dense(2, name="encoder_output")(x)

#5. O Model keras define o encoder - modelo que recebe uma entrada uma imagem (32,32,1) e codifica em um embedding 2D (representação latente em dimensão reduzida a da entrada)
encoder = models.Model(encoder_input, encoder_output)


#Decoder
#1. define a camadas de entrada do decoder (o embedding - vetor do espeço latente)
decoder_input = layers.Input(shape=(2,), name="decoder_input")

#2. conecta a entrada a camada densa
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)

#3. Faz o reshape do vetor em um tensor que pode ser fornecido como entrada para primeira camada Convolucional
x = layers.Reshape(shape_before_flattening)(x)

#4. Empilha camadas convolucionaios transpostas
x = layers.Conv2DTranspose(128,(3,3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(64,(3,3), strides=2, activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(32,(3,3), strides=2, activation='relu', padding='same')(x)
decoder_output = layers.Conv2D(1,(3,3), strides=1, activation='sigmoid', padding='same', name='decoder_output')(x)

#5. Modelo KERAS que define o decodificar - um modelo que recebe o embedding no espaço latente e codifica em um dominio da imagem original
decoder = models.Model(decoder_input,decoder_output)

### AUTOENCODER COMPLETO

autoencoder = models.Model(encoder_input, decoder(encoder_output))

## compilação
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

## trinamento
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=100,
    shuffle=True,
    validation_data=(x_test,x_test)
)

# Save the final models
autoencoder.save("./models/autoencoder.keras")
encoder.save("./models/encoder.keras")
decoder.save("./models/decoder.keras")

#Reconstruindo imagens - usando o modelo treinado
example_images = x_test[:5000]
predictions = autoencoder.predict(example_images)

print("Example real clothing items")
display(example_images)
print("Reconstructions")
display(predictions)



#encode nas imagens de exemplo
embeddings = encoder.predict(example_images)

# alguns exemplos de embeddings
print(embeddings[:10])

#plot
plt.figure(figsize=(8,8))
plt.scatter(embeddings[:,0], embeddings[:,1], c="black", alpha=0.5, s=3)
plt.show()

# Colorindo embeddings de acordo com o label
example_labels = y_test[:5000]

figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    cmap="rainbow",
    c=example_labels,
    alpha=0.8,
    s=3,
)
plt.colorbar()
plt.show()

mins, maxs = np.min(embeddings, axis=0), np.max(embeddings,axis=0)
sample = np.random.uniform(mins,maxs,size=(18,2))
reconstructions = decoder.predict(sample)


grid_width, grid_height = (6, 3)


figsize = 8
plt.figure(figsize=(figsize, figsize))


plt.scatter(embeddings[:, 0], embeddings[:, 1], c="black", alpha=0.5, s=2)


plt.scatter(sample[:, 0], sample[:, 1], c="#00B0F0", alpha=1, s=40)
plt.show()

# Add underneath a grid of the decoded images
fig = plt.figure(figsize=(figsize, grid_height * 2))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
    ax = fig.add_subplot(grid_height, grid_width, i + 1)
    ax.axis("off")
    ax.text(
        0.5,
        -0.35,
        str(np.round(sample[i, :], 1)),
        fontsize=10,
        ha="center",
        transform=ax.transAxes,
    )
    ax.imshow(reconstructions[i, :, :], cmap="Greys")