package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import java.io.{File, FileInputStream}
import java.util

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import com.intel.analytics.zoo.pipeline.inference.{InferenceSupportive, JTensor}
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.io.Source

object Tests extends App with InferenceSupportive {

  ////////////loading images
  val imageFolder = new File("/home/joy/workspace/images").listFiles.toList
  //val imageFolder = new File("/home/joy/workspace/mobImage").listFiles.toList
  //val imageFolder = new File("/home/joy/outImage").listFiles.toList
  imageFolder.foreach(println)

  val labels = Source.fromFile("/home/joy/workspace/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt").getLines.toList

 ///////OpenVINO
//  val inputs = imageFolder.map(file => {
//    val imageBytes = FileUtils.readFileToByteArray(file)
//    val imageProcess = new ImageProcessor
//    val res = imageProcess.processForOpenVINO(imageBytes, 224, 224)
//    val input = new JTensor(res, Array(224, 224, 3))
//    List(util.Arrays.asList(input)).asJava
//  })
//  inputs.foreach(println)
//
//
//
//  ///////////model_params
//  var modelType = "resnet_v1_50"
//  var checkpointPath: String = "/home/joy/workspace/models/resnet_v1_50.ckpt"
//  var ifReverseInputChannels = true
//  var inputShape = Array(1, 224, 224, 3)
//  var meanValues = Array(123.68f, 116.78f, 103.94f)
//  var scale = 1.0f
//  val fileSize = new File(checkpointPath).length()
//  val inputStream = new FileInputStream(checkpointPath)
//  val modelBytes = new Array[Byte](fileSize.toInt)
//  inputStream.read(modelBytes)
//
//  val intraOpParallelismThreads = 1
//  val interOpParallelismThreads =1
//  val usePerSessionThreads = true
//
//
//  val resnet50InferenceModel = new Resnet50OpenVINOIRInferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
//  inputs.map((input: util.List[util.List[JTensor]]) => {
//    val result = resnet50InferenceModel.doPredict(input)
//    val data = result.get(0).get(0).getData
//    val max: Float = data.max
//    val index = data.indexOf(max)
//    val label = labels(index)
//    println(index, label)
//  })
 //////////saved model ResNet --TFNet
//  val savedModelDir = "/home/joy/working/resnet50-imagenet-fc2"
//  val input = "resnet50_input"
//  val resnet50InferenceModel = new Rennet50InferenceModel2(1, savedModelDir, inputShape, ifReverseInputChannels, meanValues, scale, input)
//
//     inputs.map(input => {
//   val result = resnet50InferenceModel.doPredict(input)
//   val data = result.get(0).get(0).getData
//   val max: Float = data.max
//   val index = data.indexOf(max)
//   val label = labels(index)
//   println(index + label)
//   })
  //top is false
//  val modelPath = "/home/joy/working/resnet50-imagenet-fc1"
///////////////inputs
  val inputs = imageFolder.map(file => {
    val imageBytes = FileUtils.readFileToByteArray(file)
    val imageProcess = new ImageProcessor
    val res: Array[Float] = imageProcess.processForTFNetMobileNet(imageBytes, 224, 224)
    val ten = Tensor[Float](res, Array(1, 224, 224, 3))
    //val input = new JTensor(res, Array(224, 224, 3))
    //List(input).asJava
    //List(util.Arrays.asList(input)).asJava
  ten
  })
  inputs.foreach(println)
//
//  val modelinput = Array("resnet50_input:0")
//  val modelPath = "/home/joy/working/resnet50-imagenet-fc2"
//  val outputs = Array("resnet50/fc1000/Softmax:0")
//  val resnet50InferenceModel = new Rennet50TFNetInferenceModel(1, modelPath, modelinput, outputs, 1, 1, true)
//
//
//  inputs.map((input: util.List[util.List[JTensor]]) => {
//     val result = resnet50InferenceModel.doPredict(input)
//         println("result", result)
//     val data = result.get(0).get(0).getData
//         println("data", data)
//     val max: Float = data.max
//     val index = data.indexOf(max)
//     val label = labels(index)
//     println(index + label)
//     })

/////////TFNet saved Model
//  val modelPath = "/home/joy/working/tar/resnet50.tar.gz"
//  val modelInputs: Array[String] = Array("resnet50_input:0")
//  val modelOutputs = Array("resnet50/fc1000/Softmax:0")
//  val imageDir = "/home/joy/workspace/images"
//  val classesFile = "/home/joy/workspace/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt"
//  val intraOpParallelismThreads = 1
//  val interOpParallelismThreads = 1
//  val usePerSessionThreads = true
//
//  val fileSize = new File(modelPath).length()
//  val inputStream = new FileInputStream(modelPath)
//  val savedModelBytes = new Array[Byte](fileSize.toInt)
//  inputStream.read(savedModelBytes)
//
//  val resnet50InferenceModel = new Rennet50TFNetInferenceModel(1, savedModelBytes, modelInputs, modelOutputs, 1, 1, true)
//
//
//    inputs.map((input: util.List[util.List[JTensor]]) => {
//       val result = resnet50InferenceModel.doPredict(input)
//           println("result", result)
//       val data = result.get(0).get(0).getData
//           println("data", data)
//       val max: Float = data.max
//      println(max)
//       val index = data.indexOf(max)
//       val label = labels(index)
//       println(index + label)
//       })


  //val imagePath = "/home/joy/workspace/images/n04370456_5753.JPEG"
//  val imageFolder = new File("/home/joy/workspace/images").listFiles.toList
//  imageFolder.foreach(println)
//
//  val inputs = imageFolder.map(file => {
//    val imageBytes = FileUtils.readFileToByteArray(file)
//    val imageProcess = new ImageProcessor
//    val res = imageProcess.processForTFNet(imageBytes, 224, 224)
//    val input = new JTensor(res, Array(224, 224, 3))
////    val res = imageProcess.processForFrozen(imageBytes, 224, 224)
//    val input = Tensor.ones[Float](1, 224, 224, 3)
//    //val input = new JTensor(res, Array(224, 224, 3))
//    List(util.Arrays.asList(input)).asJava
//     input
//  })
//
//
//  inputImages.foreach(println)
//
  //val modelPath = "/home/joy/workspace/models/mobilenet_v1/frozen_inference_graph.pb"
  val modelPath = "/home/joy/workspace/models/mobilenet_v1"
//  //val modelPath = "/home/joy/workspace/models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
  val input = Array("input:0")
  val outputs: Array[String] = Array("MobilenetV1/Predictions/Reshape_1:0")
//
//  //val resnet50InferenceModel = new Rennet50InferenceModel(1, modelPath, inputs, outputs, 1, 1, true)
   val model = new FrozenModel(1, modelPath, 1, 1, true)
//     val modelTest = TFNet(modelPath, input, outputs)

      inputs.map((input) => {
        // val result: Tensor[Float] = modelTest.forward(input).toTensor[Float]
          val result: Tensor[Float] = model.doPredict(input).toTensor[Float]
             println("result", result)
          val data: Array[Float] = result.storage().array()
          val max = data.max
          val index = data.indexOf(max)
          println("index", index)
//         val data = result.get(0).get(0).getData
//             println("data", data)
//         val max: Float = data.max
////        println(max)
//         val index = data.indexOf(max)
         val label = labels(index-1)
         println(label)
         })

}



