import tensorflow as tf
from influxdb import InfluxDBClient
import tempfile

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting an float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=str(value)))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(values)])


def _float_feature_list(values):
    """Wrapper for inserting an float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(values)])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(values)])


def convert(raw):
    for s in raw['series']:
        # TODO: Possibly add the query as context
        # TODO: Probably add padding to last batch
        tags = s.get('tags', {})
        columns = s['columns']
        measurement = s['name']

        ctx = {"tag/" + tag: _bytes_feature(v) for tag, v in tags.items()}
        ctx["seq/measurement"] = _bytes_feature(measurement)
        ctx["seq/len"] = _int64_feature([len(s['values'])])
        context = tf.train.Features(feature=ctx)

        features = {"column/" + col: {"data": [], "type_func": None}
                    for col in columns}
        for value in s['values']:
            for col, v in zip(columns, value):
                features["column/" + col]["data"].append(v)
                if isinstance(v, float) or isinstance(v, int):
                    features["column/" + col]["type_func"] = _float_feature_list
                elif isinstance(v, int):
                   # Only update the type function if it has yet to be set.
                   # This way the float function can take priority.
                   if features["column/" + col]["type_func"] is  None:
                        features["column/" + col]["type_func"] = _int64_feature_list
                else:
                    features["column/" + col]["type_func"] = _bytes_feature_list

        feature_list = {col: f["type_func"](f["data"]) for col, f in features.items()}
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
        yield sequence_example

def query_batch(q, limit, offset, db, epoch='ms'):
    cli = InfluxDBClient()
    while True:
        qq = "{0} limit {1} offset {2}".format(q, limit, offset)
        res = cli.query(qq, database=db, epoch=epoch)
        if not res:
            break
        yield res

        offset += limit

if __name__ == "__main__":
    #cli = InfluxDBClient(ssl=True, verify_ssl=False)
    #res = cli.query("select usage_idle from cpu where time > now() - 4d group by host limit 10 offset 100", database="telegraf", epoch="ms")
    query = "select usage_idle from cpu where time > now() - 120h group by host"
              # Write all examples into a TFRecords file
    with tf.python_io.TFRecordWriter('seq.pb') as writer:
      for res in query_batch(query, limit=100, offset=100, db="telegraf"):
          for s in convert(res.raw):
            writer.write(s.SerializeToString())
