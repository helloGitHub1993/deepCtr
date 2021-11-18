import tensorflow as tf
from layer.common_layer import dnn,cross,get_embedding

class Dcn():
    def __init__(self,l2_reg_linear=0.00001,dcn_cross_layers=2,dnn_layers=[32,32],linear_deep_init_size=20,task='binary'):
        self.featurn_num = 17
        self.embedding_size = 4
        self.data = tf.placeholder(tf.int32,[None,self.featurn_num])
        self.label = tf.placeholder(tf.float32,[None,])
        self.lr = tf.placeholder(tf.float32,[])
        self.feature_dict = {"ret_num":8,
                            "cmt_num":8,
                            "like_num":8,
                            "act_num":8,
                            "interact_num":8,
                            "expo_num":8,
                            "recent_ret_num":8,
                            "recent_cmt_num":8,
                            "recent_like_num":8,
                            "recent_act_num":8,
                            "recent_real_expo_num":8,
                            "effect_weight":8,
                            "author_followers_num":8,
                            "click_rate":8,
                            "interact_rate":8,
                            "click_rate_recent":8,
                            "interact_rate_recent":8}

        self.feature_list = ["ret_num",
                            "cmt_num",
                            "like_num",
                            "act_num",
                            "interact_num",
                            "expo_num",
                            "recent_ret_num",
                            "recent_cmt_num",
                            "recent_like_num",
                            "recent_act_num",
                            "recent_real_expo_num",
                            "effect_weight",
                            "author_followers_num",
                            "click_rate",
                            "interact_rate",
                            "click_rate_recent",
                            "interact_rate_recent"]

        self.data_embedding = get_embedding(self.feature_list,self.feature_dict,self.data,self.embedding_size,len(self.feature_dict))

        self.data_embedding = tf.reshape(self.data_embedding,[-1, self.embedding_size*len(self.feature_dict)])        

        #dnn part
        dnn_out = dnn(self.data_embedding,dnn_layers)
        #dcn_part
        dcn_out = cross(self.data_embedding,dcn_cross_layers)

        out = tf.concat([dnn_out,dcn_out],axis=1)
        #out = dcn_out

        self.logits = tf.layers.dense(inputs=out,units=1,activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_linear),name="fullc_3")
        self.logits = tf.reshape(self.logits,[-1,])
        self.predict = tf.nn.sigmoid(self.logits)

        #step variable
        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.global_epoch_step = tf.Variable(0,trainable=False,name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step,self.global_epoch_step+1)

        regulation_rate = 0.0
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.label))

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        #self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss,trainable_params)
        clip_gradients,_ = tf.clip_by_global_norm(gradients,5)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients,trainable_params),global_step=self.global_step)

    def train(self,sess,train_data,label,lr):
        loss,_ = sess.run([self.loss,self.train_op],feed_dict={self.data:train_data,self.label:label,self.lr:lr})
        return loss

    def _eval(self,sess,data):
        predict = sess.run(self.predict,feed_dict={self.data:data})
        return predict

    def save(self,sess,path):
        saver = tf.train.Saver()
        saver.save(sess,save_path=path)

    def restore(self,sess,path):
        saver = tf.train.Saver()
        saver.restore(sess,save_path=path)
