import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings
from tensorflow.keras import Sequential, Model, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,Lambda,Input,Activation,Dropout
from tensorflow.keras.optimizers import Adam
from Utils import  get_all_data
import matplotlib.pyplot as pl
from sklearn import metrics
from tensorflow.keras import regularizers
from tqdm import tqdm
from Config import Config
from tqdm import tqdm
from openpyxl import Workbook,load_workbook
from _CAAM import MDD
from _CAAM.utils import accuracy
from CPAC_Models import create_feature_extractor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)

    
def _get_encoder(input_shape,num_classes=len(Config.CLASS_LABELS)):
    return create_feature_extractor(input_shape,num_classes)

def _get_temporal_encoder(input_shape,num_classes=len(Config.CLASS_LABELS)):
    return create_temporal_extractor(input_shape,num_classes)

def _get_task(input_shape=(64,),num_classes=len(Config.CLASS_LABELS)):
    inputs = Input(shape = (input_shape[0],))
    ac = Activation('relu')(inputs)
    drop = Dense(num_classes,activation='softmax')(ac)#
    model = Model(inputs = inputs,outputs = drop)
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.001,beta_1=0.975, beta_2=0.932,epsilon=1e-8), metrics = ['accuracy'])
    print("Classifier create success!")
    return model

def _get_discriminator(input_shape=(64,),num_classes=len(Config.CLASS_LABELS)):
    inputs = Input(shape = (input_shape[0],))
    ac = Activation('relu')(inputs)
    drop = Dense(num_classes,activation='softmax')(ac)#
    model = Model(inputs = inputs,outputs = drop)
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.001,beta_1=0.975, beta_2=0.932,epsilon=1e-8), metrics = ['accuracy'])
    print("Discriminator create success!")
    return model

#定义基本信息
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad")
DATA_PATH_casia = 'CASIA_transfer_96'
DATA_PATH_emodb = 'EMODB_transfer_96'
DATA_PATH_savee = 'SAVEE_transfer_96'
DATA_PATH_ravde = 'RAVDE_transfer_96'
DATA_PATH_emodb_e = 'EMODB_transfer_96_enhance'
DATA_PATH_savee_e = 'SAVEE_transfer_96_enhance'

#提取MFCC特征
x_casia, y_casia= get_all_data(DATA_PATH_casia, class_labels = CLASS_LABELS, flatten = False)
x_emodb, y_emodb= get_all_data(DATA_PATH_emodb, class_labels = CLASS_LABELS, flatten = False)
# x_savee, y_savee= get_all_data(DATA_PATH_savee, class_labels = CLASS_LABELS, flatten = False)
# x_ravde, y_ravde= get_all_data(DATA_PATH_ravde, class_labels = CLASS_LABELS, flatten = False)

seed = 45
tf.random.set_seed(seed)#C2B 44 B2C 37
np.random.seed(seed)
data = {"CASIA":(x_casia, y_casia,CLASS_LABELS),"EMODB":(x_emodb, y_emodb,CLASS_LABELS)}
        # "SAVEE":(x_savee, y_savee,CLASS_LABELS),"RAVDE":(x_ravde, y_ravde,CLASS_LABELS)}
data_name = ["CASIA","EMODB"]#,"SAVEE"]
colors=['#235389','#0fb5e8']
used_name = []

data_df = {"Instance":["TEST"],"UAR":["TEST"],"WAR":["TEST"]}
data_df = pd.DataFrame(data_df)
pw = pd.ExcelWriter('MDD.xlsx')
for source_name in data_name:
    for target_name in data_name:
        if source_name==target_name:
            continue
        print(source_name+" transfer to "+target_name+":")
        x_source = data[source_name][0]
        x_source = x_source.reshape(x_source.shape[0], x_source.shape[1], x_source.shape[2], 1)
        y_source = data[source_name][1]
        y_source = to_categorical(y_source,num_classes=len(CLASS_LABELS))
        
        x_target = data[target_name][0]
        x_target = x_target.reshape(x_target.shape[0], x_target.shape[1], x_target.shape[2], 1)
        y_target = data[target_name][1]
        y_target = to_categorical(y_target,num_classes=len(CLASS_LABELS))
        
        model = MDD(_get_encoder(x_source.shape[1:]), _get_task(),_get_discriminator(),loss="categorical_crossentropy", 
            optimizer=Adam(learning_rate=0.01,beta_1=0.975, beta_2=0.935,epsilon=1e-8), metrics=["accuracy"])
        
        weight_path = './Models/'+source_name+"_2_"+target_name+"_MDD_weight.hdf5"
        checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='loss',save_weights_only=True,save_best_only=False, verbose=0,mode='min')
        for i in tqdm(range(120)):
            model.fit(x_source, y_source, x_target, y_target,epochs=1, batch_size=512, verbose=0, callbacks=[checkpoint])
        
        print("UAR_target={}".format(metrics.recall_score(np.argmax(y_target,axis=-1), np.argmax(model.predict(x_target),axis=-1), average='macro')))
        print("WAR_target={}".format(metrics.recall_score(np.argmax(y_target,axis=-1), np.argmax(model.predict(x_target),axis=-1), average='weighted')))
        print("UAR_Source={}".format(metrics.recall_score(np.argmax(y_source,axis=-1), np.argmax(model.predict(x_source),axis=-1), average='macro')))
        print("WAR_Source={}".format(metrics.recall_score(np.argmax(y_source,axis=-1), np.argmax(model.predict(x_source),axis=-1), average='weighted')))
data_df.to_excel(pw,index=False)
pw.save()        