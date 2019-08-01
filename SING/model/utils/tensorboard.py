import tensorflow as tf

def log_stuff_to_tensorboard(step, loss, grads):
	tf.summary.scalar('Conv_AE_Loss', tf.reduce_sum(loss), step=int(ckpt.step))
