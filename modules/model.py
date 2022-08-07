import tensorflow as tf
from keras.models import Model
from keras import regularizers
from keras.layers.core import Dense, Dropout
from keras.layers import Input, ELU, CuDNNLSTM, Bidirectional, Embedding, Convolution1D, MaxPooling1D, concatenate
from keras.optimizers import Adam
from modules.Attention_layer import Attention_layer

import numpy as np

import warnings
warnings.filterwarnings("ignore")

class model:
    def __init__(self, flg = False):
        self.epochs = 10             #epochs
        self.batch_size = 64         #batch
        self.lstm_output_size=128    #LSTM Unit
        self.embedding_dim=128       #dimension
        self.lr=1e-3                 #Learning Rate
        self.kernel_size=128           #CNN kernel_size
        self.filters=256             #CNN filters
        self.pool_size=3            #CNN pool_size
        self.model_url_path = 'model/url.h5'
        self.model_html_path = 'model/html.h5'
        self.model_dns_path = 'model/dns.h5'
        self.web2vec_path = 'model/web2vec.h5'
        self.trigger = 255
        if flg == True:
            self.poison_url_path = 'model/poison_url.h5'
            self.poison_html_path = 'model/poison_html.h5'
            self.poison_dns_path = 'model/poison_dns.h5'
            self.poison_web2vec_path = 'model/poison_web2vec.h5'
        self.path = "/home/Yuji/m1/phish/data/"

    def web2vec(self, c_sequence_length_url, c_vocabulary_size_url, 
                    w_vocabulary_size_url, w_sequence_length_url, 
                    sequence_length_dom, sequence_length_html_w, sequence_length_sent,
                    vocabulary_size_dom, vocabulary_size_html_w, vocabulary_size_sent,
                    c_sequence_length_dns, c_vocabulary_size_dns, sequence_length_dns,
                    vocabulary_size_dns, W_reg=regularizers.l2(1e-4)):
        # Input url_char
        input_url_char = Input(shape=(c_sequence_length_url,), dtype='int32', name='url_char_input')
        # Embedding layer
        emb_url_char = Embedding(input_dim=c_vocabulary_size_url, output_dim=self.embedding_dim, input_length=c_sequence_length_url,W_regularizer=W_reg)(input_url_char)
        #input url_word
        input_url_word = Input(shape=(w_sequence_length_url,), dtype='int32', name='url_word_input')
        # Embedding layer
        emb_url_word = Embedding(input_dim=w_vocabulary_size_url, output_dim=self.embedding_dim, input_length=w_sequence_length_url,W_regularizer=W_reg)(input_url_word)

        #url_char_model
        emb_url_char = Dropout(0.5)(emb_url_char)
        conv1 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_url_char)
        conv1 = ELU()(conv1)
        conv1 = MaxPooling1D(pool_size=self.pool_size)(conv1)
        conv1 = Dropout(0.5)(conv1)
        lstm1 = Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv1)

        lstm1= Dropout(0.5)(lstm1)
        lstm1 = Attention_layer()(lstm1)

        # url_word_model

        emb_url_word = Dropout(0.5)(emb_url_word)
        conv2 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_url_word)
        conv2 = ELU()(conv2)
        conv2 = MaxPooling1D(pool_size=self.pool_size)(conv2)
        conv2 = Dropout(0.5)(conv2)
        lstm2 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv2)

        lstm2 = Dropout(0.5)(lstm2)
        lstm2 = Attention_layer()(lstm2)

        #concatenate
        x_url_output = concatenate([lstm1, lstm2], axis=1)
        #x_url_output = Dense(128, activation='relu')(x_url_output)


        #DOM model
        input_dom = Input(shape=(sequence_length_dom,), dtype='int32', name='dom_input')
        # Embedding layer
        emb_dom = Embedding(input_dim=vocabulary_size_dom, output_dim=self.embedding_dim, input_length=sequence_length_dom,W_regularizer=W_reg)(input_dom)
        emb_dom = Dropout(0.5)(emb_dom)
        # Conv layer
        conv3 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dom)
        conv3 = ELU()(conv3)
        conv3 = MaxPooling1D(pool_size=self.pool_size)(conv3)
        conv3 = Dropout(0.5)(conv3)
        # LSTM layer
        lstm3 = Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv3)

        lstm3 = Dropout(0.5)(lstm3)
        lstm3 = Attention_layer()(lstm3)


        #text__word
        input_text_word = Input(shape=(sequence_length_html_w,), dtype='int32', name='text_word_input')
        # Embedding layer
        emb_text_word = Embedding(input_dim=vocabulary_size_html_w, output_dim=self.embedding_dim, input_length=sequence_length_html_w,W_regularizer=W_reg)(input_text_word)
        emb_text_word = Dropout(0.5)(emb_text_word)

        conv4 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_text_word)
        conv4 = ELU()(conv4)
        conv4 = MaxPooling1D(pool_size=self.pool_size)(conv4)
        conv4 = Dropout(0.5)(conv4)
        lstm4 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv4)

        lstm4 = Dropout(0.5)(lstm4)
        lstm4 = Attention_layer()(lstm4)

        #text_sentence
        input_text_sent = Input(shape=(sequence_length_sent,), dtype='int32', name='text_sent_input')
        # Embedding layer
        emb_text_sent = Embedding(input_dim=vocabulary_size_sent, output_dim=self.embedding_dim, input_length=sequence_length_sent,
                            W_regularizer=regularizers.l2(1e-4))(input_text_sent)

        emb_text_sent=Dropout(0.5)(emb_text_sent)

        # Conv layer
        conv5 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_text_sent)
        conv5 = ELU()(conv5)
        conv5 = MaxPooling1D(pool_size=self.pool_size)(conv5)
        conv5 = Dropout(0.5)(conv5)
        # LSTM layer
        # lstm5 = Bidirectional( LSTM(self.lstm_output_size, return_sequences=True))(conv5)
        lstm5 = CuDNNLSTM(self.lstm_output_size,return_sequences=True)(conv5)

        lstm5 = Dropout(0.5)(lstm5)
        lstm5 = Attention_layer()(lstm5)
        
        x_text_output = concatenate([lstm4, lstm5], axis=1)
        
        
        #x_text_output = Dense(128, activation='relu')(x_text_output)
        input_dns_char = Input(shape=(c_sequence_length_dns,), dtype='int32', name='dns_char_input')
        # Embedding layer
        emb_dns_char = Embedding(input_dim=c_vocabulary_size_dns, output_dim=self.embedding_dim, 
                                    input_length=c_sequence_length_dns,W_regularizer=regularizers.l2(1e-4))(input_dns_char)
        #input dns_word
        input_dns_word = Input(shape=(sequence_length_dns,), dtype='int32', name='dns_word_input')
        # Embedding layer
        emb_dns_word = Embedding(input_dim=vocabulary_size_dns, output_dim=self.embedding_dim, 
                                    input_length=sequence_length_dns,W_regularizer=regularizers.l2(1e-4))(input_dns_word)
        #dns_char_model
        emb_dns_char = Dropout(0.5)(emb_dns_char)
        conv6 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dns_char)
        conv6 = ELU()(conv6)
        conv6 = MaxPooling1D(pool_size=self.pool_size)(conv6)
        conv6 = Dropout(0.5)(conv6)
        lstm1 =Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv6)
        lstm1= Dropout(0.5)(lstm1)
        lstm1 = Attention_layer()(lstm1)
        # dns_word_model
        emb_dns_word = Dropout(0.5)(emb_dns_word)
        conv7 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dns_word)
        conv7 = ELU()(conv7)
        conv7 = MaxPooling1D(pool_size=self.pool_size)(conv7)
        conv7 = Dropout(0.5)(conv7)
        lstm2 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv7)
        lstm2 = Dropout(0.5)(lstm2)
        lstm2 = Attention_layer()(lstm2)        
        #concatenate
        x_dns_output = concatenate([lstm1, lstm2], axis=1)


        x=concatenate([x_url_output,lstm3,x_text_output, x_dns_output],axis=1)
        #x=Flatten()(x)
        x=Dense(256,activation='relu')(x)
        x=Dense(128, activation='relu')(x)
        x=Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)

        # Compile model and define optimizer
        model = Model(input=[input_url_char,input_url_word,input_dom,input_text_word,input_text_sent, input_dns_char, input_dns_word], output=[output])

        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def url_lstm(self, c_sequence_length_url, c_vocabulary_size_url, 
                    w_vocabulary_size_url, w_sequence_length_url, W_reg=regularizers.l2(1e-4)):
        
        input_url_char = Input(shape=(c_sequence_length_url,), dtype='int32', name='url_char_input')
        # Embedding layer
        emb_url_char = Embedding(input_dim=c_vocabulary_size_url, output_dim=self.embedding_dim, input_length=c_sequence_length_url,W_regularizer=W_reg)(input_url_char)
        #input url_word
        input_url_word = Input(shape=(w_sequence_length_url,), dtype='int32', name='url_word_input')
        # Embedding layer
        emb_url_word = Embedding(input_dim=w_vocabulary_size_url, output_dim=self.embedding_dim, input_length=w_sequence_length_url,W_regularizer=W_reg)(input_url_word)

        #url_char_model
        emb_url_char = Dropout(0.5)(emb_url_char)
        conv1 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_url_char)
        conv1 = ELU()(conv1)
        conv1 = MaxPooling1D(pool_size=self.pool_size)(conv1)
        conv1 = Dropout(0.5)(conv1)

        lstm1 = Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv1)
        lstm1= Dropout(0.5)(lstm1)
        lstm1 = Attention_layer()(lstm1)

        # url_word_model

        emb_url_word = Dropout(0.5)(emb_url_word)
        conv2 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_url_word)
        conv2 = ELU()(conv2)
        conv2 = MaxPooling1D(pool_size=self.pool_size)(conv2)
        conv2 = Dropout(0.5)(conv2)

        lstm2 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv2)
        lstm2 = Dropout(0.5)(lstm2)
        lstm2 = Attention_layer()(lstm2)

        #concatenate
        x_url_output = concatenate([lstm1, lstm2], axis=1)
        #x_url_output = Dense(128, activation='relu')(x_url_output)

        #concatenate
        x = concatenate([lstm1, lstm2], axis=1)
        x=Dense(256,activation='relu')(x)
        x=Dense(128, activation='relu')(x)
        x=Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)

        # Compile model and define optimizer
        model = Model(input=[input_url_char,input_url_word], output=[output])

        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model


###########
##　2段階 ##
###########
    def dns_lstm(self, c_sequence_length_dns, c_vocabulary_size_dns, sequence_length_dns, vocabulary_size_dns):
        # Input dns_char
        input_dns_char = Input(shape=(c_sequence_length_dns,), dtype='int32', name='dns_char_input')
        # Embedding layer
        emb_dns_char = Embedding(input_dim=c_vocabulary_size_dns, output_dim=self.embedding_dim, 
                                 input_length=c_sequence_length_dns,W_regularizer=regularizers.l2(1e-4))(input_dns_char)
        #input dns_word
        input_dns_word = Input(shape=(sequence_length_dns,), dtype='int32', name='dns_word_input')
        # Embedding layer
        emb_dns_word = Embedding(input_dim=vocabulary_size_dns, output_dim=self.embedding_dim, 
                                 input_length=sequence_length_dns,W_regularizer=regularizers.l2(1e-4))(input_dns_word)
        #dns_char_model
        emb_dns_char = Dropout(0.5)(emb_dns_char)
        conv1 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dns_char)
        conv1 = ELU()(conv1)
        conv1 = MaxPooling1D(pool_size=self.pool_size)(conv1)
        conv1 = Dropout(0.5)(conv1)
        lstm1 =Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv1)
        lstm1= Dropout(0.5)(lstm1)
        lstm1 = Attention_layer()(lstm1)
        # dns_word_model
        emb_dns_word = Dropout(0.5)(emb_dns_word)
        conv2 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dns_word)
        conv2 = ELU()(conv2)
        conv2 = MaxPooling1D(pool_size=self.pool_size)(conv2)
        conv2 = Dropout(0.5)(conv2)
        lstm2 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv2)
        lstm2 = Dropout(0.5)(lstm2)
        lstm2 = Attention_layer()(lstm2)        
        #concatenate
        x = concatenate([lstm1, lstm2], axis=1)
        #x=Flatten()(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)
        # Compile model and define optimizer
        model = Model(input=[input_dns_char,input_dns_word], output=[output])
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def html_lstm(self, sequence_length_dom, sequence_length_html_w, sequence_length_sent,
                    vocabulary_size_dom, vocabulary_size_html_w, vocabulary_size_sent, 
                    W_reg=regularizers.l2(1e-4)):
        #DOM model
        input_dom = Input(shape=(sequence_length_dom,), dtype='int32', name='dom_input')
        # Embedding layer
        emb_dom = Embedding(input_dim=vocabulary_size_dom, output_dim=self.embedding_dim, input_length=sequence_length_dom,W_regularizer=W_reg)(input_dom)
        emb_dom = Dropout(0.5)(emb_dom)
        # Conv layer
        conv3 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_dom)
        conv3 = ELU()(conv3)
        conv3 = MaxPooling1D(pool_size=self.pool_size)(conv3)
        conv3 = Dropout(0.5)(conv3)
        # LSTM layer
        lstm3 = Bidirectional(CuDNNLSTM(self.lstm_output_size,return_sequences=True))(conv3)

        lstm3 = Dropout(0.5)(lstm3)
        lstm3 = Attention_layer()(lstm3)


        #text__word
        input_text_word = Input(shape=(sequence_length_html_w,), dtype='int32', name='text_word_input')
        # Embedding layer
        emb_text_word = Embedding(input_dim=vocabulary_size_html_w, output_dim=self.embedding_dim, input_length=sequence_length_html_w,W_regularizer=W_reg)(input_text_word)
        emb_text_word = Dropout(0.5)(emb_text_word)

        conv4 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_text_word)
        conv4 = ELU()(conv4)
        conv4 = MaxPooling1D(pool_size=self.pool_size)(conv4)
        conv4 = Dropout(0.5)(conv4)
        lstm4 = Bidirectional(CuDNNLSTM(self.lstm_output_size, return_sequences=True))(conv4)

        lstm4 = Dropout(0.5)(lstm4)
        lstm4 = Attention_layer()(lstm4)

        #text_sentence
        input_text_sent = Input(shape=(sequence_length_sent,), dtype='int32', name='text_sent_input')
        # Embedding layer
        emb_text_sent = Embedding(input_dim=vocabulary_size_sent, output_dim=self.embedding_dim, input_length=sequence_length_sent,
                            W_regularizer=regularizers.l2(1e-4))(input_text_sent)

        emb_text_sent=Dropout(0.5)(emb_text_sent)

        # Conv layer
        conv5 = Convolution1D(kernel_size=self.kernel_size, filters=self.filters, border_mode='same')(emb_text_sent)
        conv5 = ELU()(conv5)
        conv5 = MaxPooling1D(pool_size=self.pool_size)(conv5)
        conv5 = Dropout(0.5)(conv5)
        # LSTM layer
        # lstm5 = Bidirectional( LSTM(self.lstm_output_size, return_sequences=True))(conv5)
        lstm5 = CuDNNLSTM(self.lstm_output_size,return_sequences=True)(conv5)

        lstm5 = Dropout(0.5)(lstm5)
        lstm5 = Attention_layer()(lstm5)

        
        x_text_output = concatenate([lstm4, lstm5], axis=1)
        #x_text_output = Dense(128, activation='relu')(x_text_output)

        x=concatenate([lstm3,x_text_output],axis=1)
        #x=Flatten()(x)
        x=Dense(256,activation='relu')(x)
        x=Dense(128, activation='relu')(x)
        x=Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid', name='output')(x)

        # Compile model and define optimizer
        model = Model(input=[input_dom,input_text_word,input_text_sent], output=[output])

        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model


    
    def predict_pt1(self, model_url, model_html, model_dns, thr, data_url, data_html, data_dns):
        prediction = model_url.predict(data_url)
        if 1 - thr >= prediction and prediction >= thr:
            prediction = model_dns.predict(data_dns)
            if 1 - thr >= prediction and prediction>= thr:
                prediction = model_html.predict(data_html)
        return prediction