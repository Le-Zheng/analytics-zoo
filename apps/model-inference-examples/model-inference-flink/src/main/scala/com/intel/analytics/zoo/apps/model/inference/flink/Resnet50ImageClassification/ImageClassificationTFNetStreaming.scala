package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

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
object ImageClassificationTFNetStreaming {

  def main(args: Array[String]): Unit = {

    // Define parameters
    // Define and obtain arguments from Params
    val savedModelPath = "/home/joy/working/tar/resnet50.tar.gz"
    val modelInputs: Array[String] = Array("resnet50_input:0")
    val modelOutputs = Array("resnet50/fc1000/Softmax:0")
    val imageDir = "/home/joy/workspace/images"
    val classesFile = "/home/joy/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt"
    val intraOpParallelismThreads = 1
    val interOpParallelismThreads = 1
    val usePerSessionThreads = true

//    try {
//      val params = ParameterTool.fromArgs(args)
//      savedModelPath = params.get("modelPath")
//      modelInputs = Array(params.get("modelInputs"))
//      modelOutputs = Array(params.get("modelOutputs"))
//      imageDir = params.get("image")
//      classesFile = params.get("classes")
//      intraOpParallelismThreads = if (params.has("intraOpParallelismThreads")) params.getInt("intraOpParallelismThreads") else 1
//      interOpParallelismThreads = if (params.has("interOpParallelismThreads")) params.getInt("interOpParallelismThreads") else 1
//      usePerSessionThreads = if (params.has("usePerSessionThreads")) params.getBoolean("usePerSessionThreads") else true
//    } catch {
//      case e: Exception => {
//        System.err.println("Please run 'ImageClassificationTFNetStreaming --modelPath <modelPath> --modelInputs <modelInputs> --modelOutputs <modelOutputs> " +
//          "--imageDir <imageDir> --classesFile <classesFile> --intraOpParallelismThreads <intraOpParallelismThreads>  --interOpParallelismThreads <interOpParallelismThreads> --usePerSessionThreads <usePerSessionThreads>")
//        return
//      }
//    }

    println("params resolved", savedModelPath, modelInputs.mkString, modelOutputs.mkString, imageDir, classesFile, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)

    // Define savedModelBytes for model streaming on Flink
    val fileSize = new File(savedModelPath).length()
    val inputStream = new FileInputStream(savedModelPath)
    val savedModelBytes = new Array[Byte](fileSize.toInt)
    inputStream.read(savedModelBytes)

    val labels: List[String] = Source.fromFile(classesFile).getLines.toList

    // Image loading and pre-processing
    // Load images from folder, and hold images as a list
    val fileList = new File(imageDir).listFiles.toList
    println("ImageList", fileList)

    // Image pre-processing
    val inputImages= fileList.map(file => {
      // Read image as Array[Byte]
      val imageBytes = FileUtils.readFileToByteArray(file)
      // Execute image processing with ImageProcessor class
      val imageProcess = new ImageProcessor
      val res = imageProcess.processForTFNetResNet(imageBytes, 224, 224)
      // Convert to required List[List[JTensor]]] input
      val input = new JTensor(res, Array(1, 224, 224, 3))
      List(util.Arrays.asList(input)).asJava
    })

    println("start ImageClassificationStreaming job...")

    // Getting started the Flink Program
    // Obtain a Flink execution environment
    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
    //val env = StreamExecutionEnvironment.createLocalEnvironment()

    // Create and transform DataStreams
    val dataStream: DataStream[JList[JList[JTensor]]] = env.fromCollection(inputImages)
    // val labelStream: DataStream[String] = env.fromCollection(labels)

    // Specify the transformation functions
    // First define an Analytics Zoo InferenceModel class to load the pre-trained model. And specify the map fucntion with InferenceModel predict function.
    val resultStream = dataStream.map(new ModelPredictionMapFunctionTFNet(savedModelBytes, modelInputs, modelOutputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads))

    //.map(result => labels(result))
    //resultStream.map{x =>labels(x)}
    //val res: DataStreamSink[Int] = resultStream.print()

    resultStream.writeAsText("/home/joy/out").setParallelism(1)

    //println(labels(res.toString().toInt))
    //System.out.println(resultStream)

    // Trigger the program execution on Flink
    env.execute("ImageClassificationTFNetStreaming")
    //System.out.println(res)

    // Collect final results, and print predicted classes

    // val results: Unit = DataStreamUtils.collect(resultStream.javaStream).asScala.foreach((i) => println(labels(i)))
//    println("Printing result ...")
//    results.foreach(println)
//    results.foreach((i) => println(labels(i)))
  }
}

// Define map fucntion class which extends RichMapFunction
class ModelPredictionMapFunctionTFNet(savedModelBytes: Array[Byte], inputs: Array[String], outputs: Array[String], intraOpParallelismThreads: Int, interOpParallelismThreads: Int, usePerSessionThreads: Boolean) extends RichMapFunction[JList[JList[JTensor]], Int] {
  var resnet50InferenceModel: Rennet50TFNetInferenceModel = null

  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Rennet50TFNetInferenceModel(1, savedModelBytes, inputs, outputs, intraOpParallelismThreads, interOpParallelismThreads, usePerSessionThreads)
  }

  override def close(): Unit = {
    resnet50InferenceModel.doRelease()
  }

  // Define map function
  override def map(in: JList[JList[JTensor]]): (Int) = {
    val outputData = resnet50InferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    System.out.println(index)
   (index)
  }
}
