	3333???@3333???@!3333???@	ד?ifA@ד?ifA@!ד?ifA@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$3333???@?"??~j??A??~j??@YD?l?{5?@*	    ?0A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?z?G5?@!???8??X@)?z?G5?@1???8??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???S5?@!1,???X@)?~j?t???1<????Q?:Preprocessing2F
Iterator::Model1?Z5?@!      Y@)?~j?t?x?1<????A?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9ד?ifA@I}?$?LP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?"??~j???"??~j??!?"??~j??      ??!       "      ??!       *      ??!       2	??~j??@??~j??@!??~j??@:      ??!       B      ??!       J	D?l?{5?@D?l?{5?@!D?l?{5?@R      ??!       Z	D?l?{5?@D?l?{5?@!D?l?{5?@b      ??!       JCPU_ONLYYד?ifA@b q}?$?LP@