package com.intel.analytics.zoo.apps.model.inference.flink.ImageClassification

import java.io.{File, FileInputStream}
import java.{io, util}
import java.util.{List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.commons.io.FileUtils
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream
import org.apache.flink.streaming.api.datastream.{DataStreamSink, DataStreamUtils}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}

import scala.collection.JavaConverters._
import scala.io.Source

object ImageClassificationStreaming {

  def main(args: Array[String]): Unit = {

    // Define parameters
    // Define and obtain arguments from Params
    var modelPath = "/home/joy/workspace/models/mobilenet_v1/frozen_inference_graph.pb"
    var modelType = "frozenModel"
    var modelInputs = Array("input:0")
    var modelOutputs = Array("MobilenetV1/Predictions/Reshape_1:0")
    var imageDir = "/home/joy/workspace/images"
    var classesFile = "/home/joy/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt"
    var intraOpParallelismThreads = 1
    var interOpParallelismThreads = 1
    var usePerSessionThreads = true
    var output = "/home/joy/output"
    val params = ParameterTool.fromArgs(args)

//    try {
//      modelPath = params.get("modelPath")
//      modelType = params.get("modelType")
//      imageDir = params.get("images")
//      classesFile = params.get("classes")
//      //      modelInputs = if (params.has("modelInputs")){
//      //        modelInputsStr = params.get("modelInputs")
//      //        modelInputsStr.toArray
//      //      } else Array("input:0")
//      //      ifReverseInputChannels = if (params.has("ifReverseInputChannels")) params.getBoolean("ifReverseInputChannels") else true
//      //      meanValues = if (params.has("meanValues")) {
//      //        val meanValuesStr = params.get("meanValues")
//      //        meanValuesStr.split(",").map(_.toFloat).toArray
//      //      } else Array(123.68f, 116.78f, 103.94f)
//      //      scale = if (params.has("scale")) params.getFloat("scale") else 1.0f
//    } catch {
//      case e: Exception => {
//        System.err.println("Please run 'ImageClassificationStreaming --modelType <modelType> --checkpointPath <checkpointPath> " +
//          "--inputShape <inputShapes> --ifReverseInputChannels <ifReverseInputChannels> --imageDir <imageDir> --classesFile <classesFile> --meanValues <meanValues> --scale <scale>" +
//          "--parallelism <parallelism>'.")
//        return
//      }
//    }

    //println("params resolved", modelType, checkpointPath, imageDir, classesFile, inputShape.mkString(","), ifReverseInputChannels, meanValues.mkString(","), scale)
    println("start ImageClassificationStreaming job...")

    // ImageNet labels
    val labels = Source.fromFile(classesFile).getLines.toList

    // Image loading and pre-processing
    // Load images from folder, and hold images as a list
    val fileList = new File(imageDir).listFiles.toList
    println("ImageList", fileList)

    // Image pre-processing
    val inputImages = fileList.map(file => {
      // Read image as Array[Byte]
      val imageBytes = FileUtils.readFileToByteArray(file)
      // Execute image processing with ImageProcessor class
      val imageProcess = new ImageProcessor
      val res = imageProcess.preProcess(imageBytes, 224, 224)
      // Convert input to List[List[JTensor]]]
      val input = new JTensor(res, Array(1, 224, 224, 3))
      List(util.Arrays.asList(input)).asJava
    })

    // Getting started the Flink Program
    // Obtain a Flink execution environment
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment

    // Create and transform DataStreams
    val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputImages)

    // Specify the transformation functions
    // Before that, define class to extend InferenceModel to load the pre-trained model. And specify the map fucntion with InferenceModel predict function.
    val resultStream = dataStream.map(new ModelPredictionMapFunction(modelPath, modelType, modelInputs, modelOutputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads))

    // Obtain classfication label by index
    val results = resultStream.map(i => labels(i - 1))

    // Print results to file or stdout
    if (params.has("output")) {
      results.writeAsText(params.get("output")).setParallelism(1)
    } else {
      println("Printing result to stdout. Use --output to specify output path.");
      results.print()
    }

    // Trigger the program execution on Flink
    env.execute("ImageClassificationStreaming")
  }

}

class ModelPredictionMapFunction(modelPath: String, modelType: String, inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends RichMapFunction[JList[JList[JTensor]], Int] {
  var MobileNetInferenceModel: MobileNetInferenceModel = _

  override def open(parameters: Configuration): Unit = {
    MobileNetInferenceModel = new MobileNetInferenceModel(1, modelPath, modelType, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
  }

  override def close(): Unit = {
    MobileNetInferenceModel.doRelease()
  }

  override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = MobileNetInferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    (index)
  }
}
