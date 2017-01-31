import tensorflow as tf
import numpy as np
import threading
from influx import query_batch, convert

def queue(seq_length, input_size, output_size):
    # are used to feed data into our queue
    queue_input_data = tf.placeholder(tf.float32, shape=[seq_length, input_size])
    queue_input_target = tf.placeholder(tf.float32, shape=[seq_length, output_size])

    fifo_queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32],
                              shapes=[[input_size], [output_size]])

    enqueue_op = fifo_queue.enqueue_many([queue_input_data, queue_input_target])
    dequeue_op = fifo_queue.dequeue()

    # tensorflow recommendation:
    # capacity = min_after_dequeue + (num_threads + a small safety margin) *
    # batch_size
    data_batch, target_batch = tf.train.batch(dequeue_op, batch_size=15, capacity=40)
    return fifo_queue, queue_input_data, queue_input_target, enqueue_op, data_batch, target_batch

def influx(sess, enqueue_op, queue_input_data, queue_input_target, query, tag, column):
    for res in query_batch(query, limit=100, offset=100, db="telegraf"):
        for s in convert(res.raw):
            v = np.array(s.feature_lists.feature_list[column].feature[0].float_list.value)
            data = np.zeros((100,))
            data[:v.shape[0]] = v
            data = np.array(data).reshape([-1, 100])
            target = s.context.feature[tag]
            try:
                sess.run(enqueue_op, feed_dict={queue_input_data: data,
                                                queue_input_target: data})
            except tf.errors.CancelledError:
                break

with tf.Session() as sess:
    fifo_queue, queue_input_data, queue_input_target, enqueue_op, data_batch, target_batch = queue(seq_length=1, input_size=100, output_size=100)
    sess = tf.Session()
    query = "select usage_idle from cpu where time > now() - 120h group by host"
    tag = "tag/host"
    column = "column/usage_idle"
    enqueue_thread = threading.Thread(target=influx, args=[sess, enqueue_op, queue_input_data, queue_input_target, query, tag, column])
    enqueue_thread.isDaemon()
    enqueue_thread.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Fetch the data from the pipeline and put it where it belongs (into your
    # model)
    for i in range(5):
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
