from ops import *
import numpy as np
import os
import time


data = np.load("final_nearest_combined_v2.p")
base_data=np.load("picked_embed_v2.p")
img_label=read_data(data)

global_step = tf.Variable(0, name='global_step', trainable=False)
y = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='y')
images = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images')
z = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='z')


with tf.variable_scope(tf.get_variable_scope()) as scope:
    G = generator(z, y)
    D, D_logits = discriminator(images, y)
    D_, D_logits_ = discriminator(G, y, reuse=True)
    samples = sampler(z, y)


# sample_labels = mnist.train.labels[0:BATCH_SIZE]
values=list(base_data.values())
for _ in range(BATCH_SIZE):
    for i in range(1):                         
        values.append(list(base_data.values())[i])

sample_labels = np.asarray(values[0:64])

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))


t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
     d_optim = tf.train.AdamOptimizer(0.00005, beta1=0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
     g_optim = tf.train.AdamOptimizer(0.00005, beta1=0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,tf.train.latest_checkpoint('./results/check_point'))
    for epoch in range(3000):
        start_time = time.time()
        for i in range(int(len(data)/BATCH_SIZE)):

            batch_images = img_label[0][i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape((-3, 64, 64, 3))
            batch_labels = img_label[1][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            sess.run([d_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
            sess.run([g_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels})
            sess.run([g_optim], feed_dict={images: batch_images, z: batch_z, y: batch_labels}) # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        if epoch % 5 == 0:
            errD = d_loss.eval(feed_dict={images: batch_images, y: batch_labels, z: batch_z})
            errG = g_loss.eval({z: batch_z, y: batch_labels})
            with open('results.txt','a+') as f:
                print("epoch:[%d-%d], time_elapsed:[%4.4f]  d_loss: %.8f, g_loss: %.8f" % (epoch,i, time.time()-start_time, errD, errG),file=f)

            # generate pics
            sample = sess.run(samples, feed_dict={z: batch_z, y: sample_labels})
            samples_path = './results/pics_100/'
            save_images(sample, [8, 8], samples_path + 'epoch_%d_%d.png' % (epoch,i))
            with open('results.txt','a+') as f:
                print('save image',file=f)

            # saving check points
            checkpoint_path = os.path.join('./results/check_point/DCGAN_model_epoch_%d_.ckpt'%epoch)
            saver.save(sess, checkpoint_path, global_step=i+1)
            with open('results.txt','a+') as f:
                print('save check_point',file=f)
