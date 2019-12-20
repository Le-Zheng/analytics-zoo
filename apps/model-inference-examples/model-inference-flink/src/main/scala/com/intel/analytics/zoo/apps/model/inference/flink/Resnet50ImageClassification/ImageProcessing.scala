package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import org.opencv.core.Rect
import org.opencv.imgcodecs.Imgcodecs
import org.slf4j.LoggerFactory

trait ImageProcessing {
  val logger = LoggerFactory.getLogger(getClass)

  // convert Array[byte] to OpenCVMat.
  def byteArrayToMat(bytes: Array[Byte], imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): OpenCVMat = {
    OpenCVMethod.fromImageBytes(bytes, imageCodec)
  }

  // do a center crop by resizing a square. normalized and isclip are optional.
  def centerCrop(mat: OpenCVMat, cropWidth: Int, cropHeight: Int, normalized: Boolean = false, isClip: Boolean = false): OpenCVMat = {
    val height = mat.height().toFloat
    val width = mat.width().toFloat
    val startH = (height - cropHeight) / 2
    val startW = (width - cropWidth) / 2
    val box = BoundingBox(startW, startH, startW + cropWidth, startH + cropHeight, normalized)
    val (wStart: Float, hStart: Float, wEnd: Float, hEnd: Float) = (box.x1, box.y1, box.x2, box.y2)
    var (x1, y1, x2, y2) = if (normalized) {
      (wStart * width, hStart * height, wEnd * width, hEnd * height)
    } else {
      (wStart, hStart, wEnd, hEnd)
    }
    if (isClip) {
      x1 = Math.max(Math.min(x1, width), 0f)
      y1 = Math.max(Math.min(y1, height), 0f)
      x2 = Math.max(Math.min(x2, width), 0f)
      y2 = Math.max(Math.min(y2, height), 0f)
    }
    val rect = new Rect(x1.toInt, y1.toInt, (x2 - x1).toInt, (y2 - y1).toInt)
    val cropedMat = new OpenCVMat()
    mat.submat(rect).copyTo(cropedMat)
    Imgcodecs.imwrite("/home/joy/output.jpg", cropedMat)
    cropedMat
  }

  // convert OpenCVMat to Array
  def matToArray(mat: OpenCVMat) = {
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    data
  }

  // convert to NCHW[N,channel,height,width] Array. OpenVINO input layout is NCHW.
  def fromHWC2CHW(data: Array[Float]): Array[Float] = {
    val resArray = new Array[Float](3 * 224 * 224)
    for (h <- 0 to 223) {
      for (w <- 0 to 223) {
        for (c <- 0 to 2) {
          resArray(c * 224 * 224 + h * 224 + w) = data(h * 224 * 3 + w * 3 + c)
        }
      }
    }
    resArray
  }

  def fromCHW2HWC(data: Array[Float]): Array[Float] = {
    val resArray = new Array[Float](3 * 224 * 224)
    for (c <- 0 to 2) {
      for (h <- 0 to 223) {
        for (w <- 0 to 223) {
          resArray(h * 224 * 3 + w * 3 + c) = data(c * 224 * 224 + h * 224 + w)
        }
      }
    }
    resArray
  }

  def matToNCHWAndArray(mat: OpenCVMat) = {
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    val resArray = new Array[Float](height * width * channel)
    for (h <- 0 to height - 1) {
      for (w <- 0 to width - 1) {
        for (c <- 0 to 2) {
          resArray(c * 224 * 224 + h * 224 + w) = data(h * 224 * 3 + w * 3 + c)
        }
      }
    }
    resArray
  }

  // Normalize with channel and scale
  def channelScaledNormalize(array: Array[Float], meanR: Int, meanG: Int, meanB: Int, scale: Double) = {
    val frameLength = array.length / 3
    val channels = 3
    val mean = Array(meanR, meanG, meanB)
    var c = 0
    while (c < channels) {
      var i = 0
      while (i < frameLength) {
        val data_index = c * frameLength + i
        //println(content(data_index), ((content(data_index) - mean(c)) * scale).toFloat)
        array(data_index) = ((array(data_index) - mean(c)) * scale).toFloat
        //println(content(data_index))
        i += 1
      }
      c += 1
    }
    array
  }

  def matToNCHWAndRGBTensor(mat: OpenCVMat) = {
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    val imageTensor: Tensor[Float] = Tensor[Float]()
    imageTensor.resize(channel, height, width)
    val storage = imageTensor.storage().array()
    imageTensor.transpose(1, 2).transpose(2, 3)
    val offset = 0
    val frameLength = width * height
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = data(j * 3 + 2)
      storage(offset + j + frameLength) = data(j * 3 + 1)
      storage(offset + j + frameLength * 2) = data(j * 3)
      j += 1
    }
    imageTensor
  }

  def matToRGBTensor(mat: OpenCVMat) = {
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    val imageTensor: Tensor[Float] = Tensor[Float]()
    imageTensor.resize(channel, height, width)
    val storage = imageTensor.storage().array()
    imageTensor.transpose(1, 2).transpose(2, 3)
    val offset = 0
    val frameLength = width * height
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = data(j * 3 + 2)
      storage(offset + j + frameLength) = data(j * 3 + 1)
      storage(offset + j + frameLength * 2) = data(j * 3)
      j += 1
    }
    imageTensor
  }

  def bytesToBGRTensor(bytes: Array[Byte]): Tensor[Float] = {
    val mat = OpenCVMethod.fromImageBytes(bytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)

    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    val imageTensor: Tensor[Float] = Tensor[Float]()
    imageTensor.resize(channel, height, width)
    val storage = imageTensor.storage().array()
    imageTensor.transpose(1, 2).transpose(2, 3)
    val offset = 0
    val frameLength = width * height
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = data(j * 3)
      storage(offset + j + frameLength) = data(j * 3 + 1)
      storage(offset + j + frameLength * 2) = data(j * 3 + 2)
      j += 1
    }
    imageTensor
  }
}
