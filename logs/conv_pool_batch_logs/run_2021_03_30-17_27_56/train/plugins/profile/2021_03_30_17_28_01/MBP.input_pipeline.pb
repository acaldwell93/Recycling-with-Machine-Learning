	?$???@?$???@!?$???@	 ?#e@t~? ?#e@t~?! ?#e@t~?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?$???@????????A}?5^:??@Y?v??/??*	    (5+A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?C?l?ۋ@!??Hp??X@)?C?l?ۋ@1??Hp??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????????!??Eh??f?)????????1??Eh??f?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism/?$???!????Ks?)?? ?rh??1?p@??=_?:Preprocessing2F
Iterator::Model?~j?t???!?@?yv?)?~j?t?x?1?@?yF?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap-???ۋ@!;?ʧ?X@)????Mbp?1? ?
Lg=?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9 ?#e@t~?Iqk?.??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	}?5^:??@}?5^:??@!}?5^:??@:      ??!       B      ??!       J	?v??/???v??/??!?v??/??R      ??!       Z	?v??/???v??/??!?v??/??b      ??!       JCPU_ONLYY ?#e@t~?b qqk?.??X@