	j?t?X	?@j?t?X	?@!j?t?X	?@	!??Y#k?!??Y#k?!!??Y#k?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j?t?X	?@/?$???Ao??
	?@Y?? ?rh??*	    ^?1A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorT㥛?m?@!?%h???X@)T㥛?m?@1?%h???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch/?$???!?}?3?+]?)/?$???1?}?3?+]?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism;?O??n??!4G?? i?)???Q???1????T?:Preprocessing2F
Iterator::Model{?G?z??!?k?z?k?)????Mbp?1?"A??96?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapX9??m?@!?
?o??X@)????Mbp?1?"A??96?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9!??Y#k?I??M???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/?$???/?$???!/?$???      ??!       "      ??!       *      ??!       2	o??
	?@o??
	?@!o??
	?@:      ??!       B      ??!       J	?? ?rh???? ?rh??!?? ?rh??R      ??!       Z	?? ?rh???? ?rh??!?? ?rh??b      ??!       JCPU_ONLYY!??Y#k?b q??M???X@