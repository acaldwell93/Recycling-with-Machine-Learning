	???x?d?@???x?d?@!???x?d?@	B?#???B?#???!B?#???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???x?d?@????Mb??A?????c?@Y1?Zd??*	    ??.A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator/?$w?@!D????X@)/?$w?@1D????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch;?O??n??!???s?Im?);?O??n??1???s?Im?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?&1???!x
`???t?)???Q???1?fڵhX?:Preprocessing2F
Iterator::Model?v??/??!?G\?/w?)?~j?t?x?1????z?C?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?Vw?@!??7A??X@)????Mbp?1Q:-??:?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9A?#???I????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Mb??????Mb??!????Mb??      ??!       "      ??!       *      ??!       2	?????c?@?????c?@!?????c?@:      ??!       B      ??!       J	1?Zd??1?Zd??!1?Zd??R      ??!       Z	1?Zd??1?Zd??!1?Zd??b      ??!       JCPU_ONLYYA?#???b q????X@