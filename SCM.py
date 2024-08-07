import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,losses,optimizers
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成器模型
def build_generator(network_type):
    input_1 = layers.Input(shape=(3600, 2))
    input_2 = layers.Input(shape=(20,))
    input_3 = layers.Input(shape=(3600, 2))

    # 生成第一个向量 (1, 3600, 2)
    x1 = layers.Dense(200)(layers.Flatten()(input_1))
    x2 = layers.Dense(200)(input_2)  # Note: input_2 already has shape (10,)
    x3 = layers.Dense(200)(layers.Flatten()(input_3))
    
    x = layers.Concatenate()([x1, x2, x3])
    x = layers.Dense(400)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # efficient
    if network_type == 1:
        x = layers.Dense(200)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(100)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(50)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(1)(x)
    elif network_type == 2:
        x = layers.Dense(200)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(100)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(50)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    elif network_type == 0:
        x = layers.Dense(900)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Dense(1800)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
        x = layers.Dense(3600)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(3600 * 2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((3600, 2))(x)
    generator = models.Model(inputs=[input_1, input_2, input_3], outputs=x)
    return generator




# 判别器模型
def build_discriminator(network_type):
    # 输入第一个向量 (3600, 2)
    if network_type == 0:
        vector1_input = layers.Input(shape=(3600, 2))
    else:
        vector1_input = layers.Input(shape=(1))
    x = layers.Flatten()(vector1_input)
    x = layers.Dense(1800)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Dense(900)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(400)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(1,activation='sigmoid')(x)

    discriminator = models.Model(vector1_input, x)
    return discriminator

def create_gan(network_type=0):
    # 构建生成器和判别器
    generator = build_generator(network_type)
    discriminator = build_discriminator(network_type)

    # 编译判别器
    discriminator.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

    # 构建并编译GAN模型
    discriminator.trainable = False

    gan_input = [layers.Input(shape=(3600,2)), layers.Input(shape=(20)),layers.Input(shape=(3600,2))]
    generated_vectors = generator(gan_input)
    gan_output = discriminator(generated_vectors)

    gan = models.Model(gan_input, gan_output)
    gan.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    return gan, discriminator, generator

def train_gan(batch, real_vectors, gan, discriminator, generator, epochs=1, batch_size=128):

    for epoch in range(epochs):

        generated_vectors = generator.predict(batch)

        # 训练判别器
        d_loss_real = discriminator.train_on_batch(real_vectors, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_vectors, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        
        # 训练生成器
        g_loss = gan.train_on_batch(batch, np.ones((batch_size, 1)))

        print(f'D Loss: {d_loss[0]}, D Accuracy: {d_loss[1]}, G Loss: {g_loss}')
