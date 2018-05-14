import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sam_code(ecgdata,n_hidden_matri,learning_rate,training_epochs,batch_size):
    '''
    :param ecgdata: 心电信号
    :param n_hidden_matri: 隐层个数矩阵
    :param learning_rate: 学习率
    :param training_epochs: 训练次数
    :param batch_size: 一次训练样本数
    :return: W，b
    '''
    # 网络参数
    n_input = len(ecgdata[0])     #输入

    #tf Graph输入
    X = tf.placeholder("float", [None,n_input])

    #权重初始化
    weight_hidden_matri=np.append(n_input,n_hidden_matri)
    num_weight_hidden=len(weight_hidden_matri)
    weights={}
    for i in range(num_weight_hidden-1):
        en_key='encoder_h%d' % (i + 1)
        en_value=tf.Variable(tf.random_normal([weight_hidden_matri[i], weight_hidden_matri[i + 1]]))
        weights[en_key]=en_value
        de_key = 'decoder_h%d' % (i + 1)
        de_value = tf.Variable(tf.random_normal([weight_hidden_matri[::-1][i], weight_hidden_matri[::-1][i + 1]]))
        weights[de_key] = de_value

    #偏置值初始化
    bias_hidden_matri=np.append(n_hidden_matri[0:-1][::-1],n_input)
    num_bias_hidden=len(bias_hidden_matri)
    biases = {}
    for i in range(num_bias_hidden):
        en_key='encoder_b%d' % (i + 1)
        en_value =tf.Variable(tf.random_normal([n_hidden_matri[i]]))
        biases[en_key] = en_value
        de_key='decoder_b%d' % (i + 1)
        de_value =tf.Variable(tf.random_normal([bias_hidden_matri[i]]))
        biases[de_key] = de_value


    # 开始编码
    def encoder(x):
        #sigmoid激活函数，layer = x*weights['encoder_h1']+biases['encoder_b1']
        for i in range(len(n_hidden_matri)):
            weight_key_name = 'encoder_h%d' % (i + 1)
            bias_key_name = 'encoder_b%d' % (i + 1)
            en_value = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[weight_key_name]),
                                   biases[bias_key_name]))
            x=en_value
        return en_value

    # 开始解码
    def decoder(x):
        #sigmoid激活函数,layer = x*weights['decoder_h1']+biases['decoder_b1']
        for i in range(len(n_hidden_matri)):
            weight_key_name = 'decoder_h%d' % (i + 1)
            bias_key_name = 'decoder_b%d' % (i + 1)
            de_value = tf.nn.sigmoid(tf.add(tf.matmul(x, weights[weight_key_name]),
                                            biases[bias_key_name]))
            x = de_value
        return de_value

    # 构造模型
    encoder_op = encoder(X)
    encoder_result = encoder_op
    decoder_op = decoder(encoder_op)

    #预测
    y_pred = decoder_op
    #实际输入数据当作标签
    y_true = X

    # 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
    cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 运行Graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for end in range(batch_size, len(ecgdata), batch_size):
                begin = end - batch_size
                batch_xs = ecgdata[begin:end]
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            # 展示每次训练结果
            if epoch % 1 == 0:
                print("Epoch:", '%04d' % (epoch+1),
                    "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")
        # Applying encode and decode over test set
        #显示编码结果和解码后结果
        '''encodes = sess.run(
            encoder_result, feed_dict={X: ecgdata[:3]})
        encode_decode = sess.run(
            y_pred, feed_dict={X: ecgdata[:3]})
        # 对比原始图片重建图片
        f, a = plt.subplots(2, 3, figsize=(10, 2))
        xx=np.arange(1,501,1)
        for i in range(3):
            a[0][i].plot(xx,ecgdata[i])
            a[1][i].plot(xx,encode_decode[i])
        f.show()
        plt.draw()
        plt.waitforbuttonpress()'''
    return weights,biases
