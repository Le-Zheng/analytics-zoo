package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class Resnet50OpenVINOIRInferenceModel(var concurrentNum: Int = 1, modelType: String, checkpointBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(null, modelType, checkpointBytes, inputShape, ifReverseInputChannels, meanValues, scale)
}

class Rennet50TFNetInferenceModel(var concurrentNum: Int = 1, savedModelBytes: Array[Byte], inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(savedModelBytes, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
}


class FrozenModel(var concurrentNum: Int = 1, modelPath: String, intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(modelPath, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
}

class Rennet50InferenceModel(var concurrentNum: Int = 1, modelPath: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends InferenceModel(concurrentNum) with Serializable {
  doLoadTF(modelPath, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
}

