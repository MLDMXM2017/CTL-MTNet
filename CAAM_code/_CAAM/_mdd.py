"""
DANN
"""

import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Input, subtract,Lambda
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn import datasets, svm, metrics
from _CAAM.utils import (GradientHandler,
                         check_arrays,
                         check_network)
from _CAAM import BaseDeepFeature

EPS = K.epsilon()


def accuracy(y_true, y_pred):
    """
    Custom accuracy function which can handle
    probas vector in both binary and multi classification
    
    Parameters
    ----------
    y_true : Tensor
        True tensor.
        
    y_pred : Tensor
        Predicted tensor.
        
    Returns
    -------
    Boolean Tensor
    """
    # TODO: accuracy can't handle 1D ys.
    multi_columns_t = K.cast(K.greater(K.shape(y_true)[1], 1),
                           "float32")
    binary_t = K.reshape(K.sum(K.cast(K.greater(y_true, 0.5),
                                    "float32"), axis=-1), (-1,))
    multi_t = K.reshape(K.cast(K.argmax(y_true, axis=-1),
                             "float32"), (-1,))
    y_true = ((1 - multi_columns_t) * binary_t +
              multi_columns_t * multi_t)
    
    multi_columns_p = K.cast(K.greater(K.shape(y_pred)[1], 1),
                           "float32")
    binary_p = K.reshape(K.sum(K.cast(K.greater(y_pred, 0.5),
                                    "float32"), axis=-1), (-1,))
    multi_p = K.reshape(K.cast(K.argmax(y_pred, axis=-1),
                             "float32"), (-1,))
    y_pred = ((1 - multi_columns_p) * binary_p +
              multi_columns_p * multi_p)        
    return tf.keras.metrics.get("acc")(y_true, y_pred)

def NLLLoss(y_true,y_pred):
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    y_pred = K.one_hot(y_pred,num_classes=y_true.shape[1])
    temp = Lambda(lambda a: a[0] * a[1])([y_true, y_pred])
    # print('y_true.shape:',y_true.shape)
    loss = K.sum(temp)/y_true.shape[1]
    loss = tf.convert_to_tensor(loss)
    return loss

def categorical_squared_hinge(y_true, y_pred):
    """
    hinge with 0.5*W^2 ,SVM
    """
    y_true = 2. * y_true - 1 # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
    vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss，参考keras自带的hinge loss
#    vvv = K.square(vvvv) # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
    vv = K.sum(vvvv, 1, keepdims=False)  #axis=len(y_true.get_shape()) - 1
    v = K.mean(vv, axis=-1)
    return v

class MDD(BaseDeepFeature):
    """
    MDD: Margin Disparity Discrepancy is a feature-based domain adaptation
    method originally introduced for unsupervised classification DA.
    
    The goal of MDD is to find a new representation of the input features which
    minimizes the disparity discrepancy between the source and target domains 
    
    The discrepancy is estimated through adversarial training of three networks:
    An encoder a task network and a discriminator.
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1904.05801.pdf>`_ Y. Zhang, T. Liu, M. Long, and M. Jordan. "Bridging theory and algorithm for domain adaptation". ICML, 2019.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 is_pretrained=False,
                 lambda_=1., 
                 gamma=1.5,#[1.5 C2B] [1.6 B2C] [BS 1.3]
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 optimizer_src=None,
                 copy=True,
                 random_state=None):

        super().__init__(encoder, task, discriminator,
                         loss, metrics, optimizer, copy,
                         random_state)#初始化（会检查传入的模型，以及传入task和disc的metrics）
        self.lambda_ = lambda_
        self.gamma = gamma
        self.iter_num = 0
        
        
        
        if optimizer_src is None:
            self.optimizer_src = deepcopy(self.optimizer)
        else:
            self.optimizer_src = optimizer_src

        # self.discriminator_ = check_network(self.task_, 
        #                                copy=True,
        #                                display_name="task",
        #                                force_copy=True)
        # self.discriminator_._name = self.discriminator_._name + "_2"
        
        # self.loss_ = loss#tf.nn.softmax_cross_entropy_with_logits
        # self.loss_ = tf.keras.losses.categorical_crossentropy
        if hasattr(self.loss_, "__name__"):
            self.loss_name_ = self.loss_.__name__
        elif hasattr(self.loss_, "__class__"):
            self.loss_name_ = self.loss_.__class__.__name__
        else:
            self.loss_name_ = ""
        
        # print("MDD Initial Success!")
        

      
    def create_model(self, inputs_Xs, inputs_Xt):
        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)#构建特征提取器
        task_src = self.task_(encoded_src)#类别判别器的预测，用于和input_ys一起做损失
        task_tgt = self.task_(encoded_tgt)
        
        task_src_nograd = GradientHandler(0., name="gh_2")(task_src)#梯度不反转，
        task_tgt_nograd = GradientHandler(0., name="gh_3")(task_tgt)
        
        # TODO, add condition for bce and cce     
        #         if self.loss_name_ in ["categorical_crossentropy",
        #                                "CategoricalCrossentropy"]:

        disc_src = GradientHandler(-self.lambda_, name="gh_0")(encoded_src)#梯度反转
        disc_tgt = GradientHandler(-self.lambda_, name="gh_1")(encoded_tgt)
        disc_src = self.discriminator_(disc_src)
        disc_tgt = self.discriminator_(disc_tgt)#构建tar和src的判别器

        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       task_src_nograd=task_src_nograd,
                       task_tgt_nograd=task_tgt_nograd,
                       disc_src=disc_src,
                       disc_tgt=disc_tgt)
        # print(self.encoder_.summary())
        # print(self.task_.summary())
        # print(self.discriminator_.summary())
        # print("MDD create model success!")
        return outputs


    def get_loss(self, inputs_ys, inputs_yt, task_src,
                 task_src_nograd, task_tgt_nograd,
                 task_tgt, disc_src, disc_tgt):
        self.iter_num += 1
        task_loss =  self.loss_(inputs_ys, task_src)#inputs_ys是onehot形式
        
        disc_loss_src = self.loss_(task_src_nograd, disc_src)#task_src_nograd表示分类器的输出，也是计算MDD的目标
        
#         logloss_tgt = K.log(1.-disc_tgt)

#         disc_loss_tgt = NLLLoss(task_tgt_nograd, logloss_tgt)#task_tgt_nograd表示分类器的输出，也是计算MDD的目标
        disc_loss_tgt = self.loss_(task_tgt_nograd, 1.-disc_tgt)
        disc_loss = disc_loss_tgt + self.gamma * disc_loss_src
        
        loss = K.mean(task_loss) + K.mean(disc_loss)
        return loss


    def get_metrics(self, inputs_ys, inputs_yt,
                    task_src, task_tgt,
                    task_src_nograd, task_tgt_nograd,
                    disc_src, disc_tgt):
        #训练时输出的指标内容
        metrics = {}

        # metrics["src_Accuracy"] = categorical_accuracy(inputs_ys, task_src)
        # metrics["tar_Accuracy"] = categorical_accuracy(inputs_yt,task_tgt)
        
        task_s = self.loss_(inputs_ys, task_src)#计算分类预测结果和标签的分类损
        metrics["task_s"] = K.mean(task_s)
        task_t = self.loss_(inputs_yt, task_tgt)#如果输入标签是非空，计算目标域上的分类损失
        metrics["task_t"] = K.mean(task_t)
        
        metrics["src_disc"] = K.mean(self.gamma * self.loss_(task_src_nograd, disc_src))
        metrics["tar_disc"] = K.mean(NLLLoss(task_tgt_nograd, K.log(1.-disc_tgt)))
        metrics["disc"] = metrics["tar_disc"]+metrics["src_disc"]
        
        # names_task, names_disc = self._get_metric_names()
        
#         for metric, name in zip(self.metrics_task_, names_task):
#             metrics[name + "_s"] = metric(inputs_ys, task_src)
#             if inputs_yt is not None:
#                 metrics[name + "_t"] = metric(inputs_yt, task_tgt)
                
#         for metric, name in zip(self.metrics_disc_, names_disc):
#             metrics[name] = (metric(task_tgt_nograd, disc_tgt) -
#                 self.gamma * metric(task_src_nograd, disc_src))
        return metrics
