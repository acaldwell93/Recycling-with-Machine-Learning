	X9??~?@X9??~?@!X9??~?@	8G.]`G@8G.]`G@!8G.]`G@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X9??~?@???Q???A?&1???@Y#??~j?@*	    ??3A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch33333?@!Js????X@)33333?@1Js????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?(\?B?@!?\????X@)???Q???1????S?:Preprocessing2F
Iterator::Model?z?G?@!      Y@){?G?zt?1$???fy9?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 46.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no98G.]`G@I???Ѣ?J@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Q??????Q???!???Q???      ??!       "      ??!       *      ??!       2	?&1???@?&1???@!?&1???@:      ??!       B      ??!       J	#??~j?@#??~j?@!#??~j?@R      ??!       Z	#??~j?@#??~j?@!#??~j?@b      ??!       JCPU_ONLYY8G.]`G@b q???Ѣ?J@