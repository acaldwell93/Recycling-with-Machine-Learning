	-??o??@-??o??@!-??o??@	jM?r?jM?r?!jM?r?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$-??o??@)\???(??Ao????@Y?&1???*	    ?#A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?????T?@!?????X@)?????T?@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??~j?t??!#?!1?g?)??~j?t??1#?!1?g?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????????!I???%zo?)?~j?t?x?1????7N?:Preprocessing2F
Iterator::ModelV-???!?Ar?)????Mbp?1+m?6%D?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?A`??T?@!?????X@)????Mb`?1+m?6%4?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9jM?r?I???f??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)\???(??)\???(??!)\???(??      ??!       "      ??!       *      ??!       2	o????@o????@!o????@:      ??!       B      ??!       J	?&1????&1???!?&1???R      ??!       Z	?&1????&1???!?&1???b      ??!       JCPU_ONLYYjM?r?b q???f??X@