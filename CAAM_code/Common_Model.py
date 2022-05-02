import sys
from typing import Tuple
from Config import Config
import numpy as np
from sklearn.metrics import accuracy_score
import prettytable as pt

class Common_Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False # 模型是否已训练

    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train: 训练集样本
        y_train: 训练集标签
        x_val: 测试集样本
        y_val: 测试集标签

    '''
    def train(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError()

    '''
    predict(): 识别音频的情感

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 识别结果（标签）的list
    '''
    def predict(self, samples):
        raise NotImplementedError()
        

    '''
    predict_proba(): 音频的情感的置信概率

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 每种情感的概率
    '''
    def predict_proba(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict_proba(samples)

    '''
    save_model(): 将模型以 model_name 命名存储在 /Models 目录下
    '''
    def save_model(self, model_name: str):
        raise NotImplementedError()

    '''
    evaluate(): 在测试集上评估模型，输出准确率

    输入:
        x_test: 样本
        y_test: 标签
    '''
    def evaluate(self, x_test, y_test):
        tb = pt.PrettyTable()
        temp = ["###"]
        print(temp)
        for item in Config.CLASS_LABELS:
            temp.append(item)
        temp.append("All")
        temp.append("Correct")
        temp.append("Accuracy")
        tb.field_names = temp
        predictions = self.predict(x_test)
        y_test = np.argmax(y_test,axis=1)
        num = len(y_test)
        #情感分类统计
        emotion_num = np.zeros((20, 20), dtype=np.int)
        print(y_test)
        print(predictions)
        for i in range(num):
             emotion_num[y_test[i]][10] += 1#表示总的数量
             emotion_num[y_test[i]][predictions[i]] += 1#表示各个情感的数量
             if y_test[i]==predictions[i]:
                 emotion_num[y_test[i]][11]+=1#表示标记正确的数量
        
        for i in range(len(Config.CLASS_LABELS)):
            print(i,'类acc:',emotion_num[i][11]/emotion_num[i][10])

        print('Accuracy:%.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))
        
        
        
        for i in range(len(Config.CLASS_LABELS)):
            temp = []
            temp.append(Config.CLASS_LABELS[i])
            for j in range(len(Config.CLASS_LABELS)):
                temp.append(emotion_num[i][j])
            temp.append(emotion_num[i][10])
            temp.append(emotion_num[i][11])
            temp.append(emotion_num[i][11]/emotion_num[i][10])
            tb.add_row(temp)

        print(tb)
        '''
        predictions = self.predict(x_test)
        score = self.model.score(x_test, y_test)
        print("True Lable: ", y_test)
        print("Predict Lable: ", predictions)
        print("Score: ", score)
        '''


