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

