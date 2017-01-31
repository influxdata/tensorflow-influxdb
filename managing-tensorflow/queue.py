import tensorflow as tf
import numpy as np
import threading
from influx import query_batch, convert

def queue(seq_length, input_size, output_size):
    # are used to feed data into our queue
    queue_input_data = tf.placeholder(tf.float32, shape=[seq_length, input_size])
    queue_input_target = tf.placeholder(tf.float32, shape=[seq_length, output_size])

    fifo_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.float32, tf.float32],
                              shapes=[[input_size], [output_size]])

    enqueue_op = fifo_queue.enqueue_many([queue_input_data, queue_input_target])
    dequeue_op = fifo_queue.dequeue()

    # tensorflow recommendation:
    # capacity = min_after_dequeue + (num_threads + a small safety margin) *
    # batch_size
    data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=3, capacity=40)
    return fifo_queue, queue_input_data, queue_input_target, enqueue_op, data_batch, target_batch

def influx(sess, enqueue_op, queue_input_data, queue_input_target, db, query, column):
    for res in query_batch(query, limit=11, offset=0, db=db):
        for s in convert(res.raw):
            v = np.array(s.feature_lists.feature_list[column].feature[0].float_list.value)
            data = np.zeros((10,))
            data[:v.shape[0]-1] = v[:v.shape[0]-1]
            data = np.array(data).reshape([-1, 10]) 
            target = np.array(v[-1]).reshape([1, 1])
            try:
                sess.run(enqueue_op, feed_dict={queue_input_data: data,
                                                queue_input_target: target})
            except tf.errors.CancelledError:
                break

with tf.Session() as sess:
    fifo_queue, queue_input_data, queue_input_target, enqueue_op, data_batch, target_batch = queue(seq_length=1, input_size=10, output_size=1)
    sess = tf.Session()
    db = "tensorflowdb"
    query = "select wet_bulb_temp from qclcd where wban = '14920' and time > now() - 30d group by wban"
    column = "column/wet_bulb_temp"
    enqueue_thread = threading.Thread(target=influx, args=[sess, enqueue_op, queue_input_data, queue_input_target, db, query, column])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Fetch the data from the pipeline and put it where it belongs (into your
    # model)
    for i in range(32):
        run_options = tf.RunOptions(timeout_in_ms=4000)
        curr_data_batch, curr_target_batch = sess.run(
            [data_batch, target_batch], options=run_options)
    # shutdown everything to avoid zombies
    try:
        sess.run(fifo_queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
    except tf.errors.CancelledError:
        pass
