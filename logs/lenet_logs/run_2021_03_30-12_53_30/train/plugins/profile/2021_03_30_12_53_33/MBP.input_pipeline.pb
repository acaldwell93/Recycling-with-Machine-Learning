	/?$??@/?$??@!/?$??@	?F?	??E@?F?	??E@!?F?	??E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$/?$??@??C?l???AV-?u@Y?S㥛p@*	    hcA2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???S?p@!??Il??X@)???S?p@1??Il??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5^?Ip@!??$??X@){?G?z??1?m\??o?:Preprocessing2F
Iterator::Model㥛? p@!      Y@){?G?zt?1?m\??_?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?F?	??E@ID??fL@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??C?l?????C?l???!??C?l???      ??!       "      ??!       *      ??!       2	V-?u@V-?u@!V-?u@:      ??!       B      ??!       J	?S㥛p@?S㥛p@!?S㥛p@R      ??!       Z	?S㥛p@?S㥛p@!?S㥛p@b      ??!       JCPU_ONLYY?F?	??E@b qD??fL@