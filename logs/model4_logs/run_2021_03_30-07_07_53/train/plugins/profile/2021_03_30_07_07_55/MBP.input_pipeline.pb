	V-??]v@V-??]v@!V-??]v@	?j?:t5-@?j?:t5-@!?j?:t5-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V-??]v@)\???(??A?l???s@Y??K7?!J@*	    ?~?@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?????J@!坥T)?X@)?????J@1坥T)?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?n??J@!\ru?C?X@);?O??n??1?hG?l???:Preprocessing2F
Iterator::Model?Q??J@!      Y@)?~j?t?x?1???Q????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?j?:t5-@I?R?xQYU@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	)\???(??)\???(??!)\???(??      ??!       "      ??!       *      ??!       2	?l???s@?l???s@!?l???s@:      ??!       B      ??!       J	??K7?!J@??K7?!J@!??K7?!J@R      ??!       Z	??K7?!J@??K7?!J@!??K7?!J@b      ??!       JCPU_ONLYY?j?:t5-@b q?R?xQYU@