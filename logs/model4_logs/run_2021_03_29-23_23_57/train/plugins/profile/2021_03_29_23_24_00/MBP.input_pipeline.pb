	??? ???@??? ???@!??? ???@	?Jj??O0@?Jj??O0@!?Jj??O0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??? ???@L7?A`???A?I+Ԅ@Y`??"?=`@*	    ???@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch㥛? <`@!?????X@)㥛? <`@1?????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??~j?<`@!)?????X@)??~j?t??1}?/?s???:Preprocessing2F
Iterator::Model??/?<`@!      Y@)????Mbp?1%?k??9i?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 16.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?Jj??O0@I@me??T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L7?A`???L7?A`???!L7?A`???      ??!       "      ??!       *      ??!       2	?I+Ԅ@?I+Ԅ@!?I+Ԅ@:      ??!       B      ??!       J	`??"?=`@`??"?=`@!`??"?=`@R      ??!       Z	`??"?=`@`??"?=`@!`??"?=`@b      ??!       JCPU_ONLYY?Jj??O0@b q@me??T@