#import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

 

def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image1, in_src_image2, in_target_image = Input(shape=image_shape), Input(shape=image_shape), Input(shape=image_shape)
    merged = Concatenate()([in_src_image1, in_src_image2, in_target_image])
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1024, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(2048, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(2048, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    model = Model([in_src_image1, in_src_image2, in_target_image], patch_out)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def encoder_layer(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_layer(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g

def generator(image_shape=(256,256,1)):
    init = RandomNormal(stddev=0.02)
    in_image1, in_image2 = Input(shape=image_shape), Input(shape=image_shape)
    combined = Concatenate(axis=-1)([in_image1, in_image2])
    
    e1 = encoder_layer(combined, 128, batchnorm=False)  
    e2 = encoder_layer(e1, 256)
    e3 = encoder_layer(e2, 512)
    e4 = encoder_layer(e3, 512)
    e5 = encoder_layer(e4, 1024)
    e6 = encoder_layer(e5, 1024)
    e7 = encoder_layer(e6, 2048)
    e8 = encoder_layer(e7, 2048)
    b = Conv2D(4096, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e8)
    b = Activation('relu')(b)
    d0 = decoder_layer(b, e7, 2048)
    d1 = decoder_layer(d0, e6, 1024)
    d2 = decoder_layer(d1, e5, 1024)
    d3 = decoder_layer(d2, e4, 512, dropout=False)
    d4 = decoder_layer(d3, e3, 512, dropout=False)
    d5 = decoder_layer(d4, e2, 256, dropout=False)
    d6 = decoder_layer(d5, e1, 128, dropout=False)
    g = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d6)
    out_image = Activation('tanh')(g)
    model = Model([in_image1, in_image2], out_image)
    return model

def cgan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src1, in_src2 = Input(shape=image_shape), Input(shape=image_shape)
    gen_out = g_model([in_src1, in_src2])
    dis_out = d_model([in_src1, in_src2, gen_out]) 
    model = Model([in_src1, in_src2], [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5) #Change the learning rate based on specific needs
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100]) #Change the loss weights based on specific needs
    return model
