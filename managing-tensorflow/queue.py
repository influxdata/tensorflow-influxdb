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
    # use this to shuffle batches:
    # data_batch, target_batch = tf.train.shuffle_batch(dequeue_op, batch_size=15, capacity=40, min_after_dequeue=5)
    return fifo_queue, queue_input_data, queue_input_target, enqueue_op, data_batch, target_batch


def enqueue(sess, enqueue_op, queue_input_data, queue_input_target):
    """ Iterates over our data puts small junks into our queue."""
    under = 0
    max = len(raw_data)
    while True:
        print("starting to write into queue")
        upper = under + 20
        print("try to enqueue ", under, " to ", upper)
        if upper <= max:
            curr_data = raw_data[under:upper]
            curr_target = raw_target[under:upper]
            under = upper
        else:
            rest = upper - max
            curr_data = np.concatenate((raw_data[under:max], raw_data[0:rest]))
            curr_target = np.concatenate(
                (raw_target[under:max], raw_target[0:rest]))
            under = rest

        try:
            sess.run(enqueue_op, feed_dict={queue_input_data: curr_data,
                                            queue_input_target: curr_target})
        except tf.errors.CancelledError:
            break
        print("added to the queue")
    print("finished enqueueing")

def decode(serialized_example):
    context_features = {
        "seq/measurement": tf.VarLenFeature(dtype=tf.string),
        "seq/len": tf.FixedLenFeature([], dtype=tf.int64),
        "tag/host": tf.VarLenFeature(dtype=tf.string)
    }
    sequence_features = {
        "column/usage_idle": tf.VarLenFeature(dtype=tf.float32),
        "column/time": tf.VarLenFeature(dtype=tf.int64)
    }

    ctx, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    seq_len = tf.cast(ctx["seq/len"], tf.int32)
    return features['column/usage_idle'], ctx['tag/host']

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
