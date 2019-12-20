package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import com.intel.analytics.bigdl.tensor.Tensor


class ImageProcessor extends ImageProcessing {

  def processForOpenVINO(bytes: Array[Byte], cropWidth: Int, cropHeight: Int) = {
    val imageMat = byteArrayToMat(bytes)
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    // The OpenVINO backend accepts input shape of NCHW[N,channel,height,width]
    val imageArray = matToNCHWAndArray(imageCent)
    imageArray
  }

  def processForTFNetMobileNet(bytes: Array[Byte], cropWidth: Int, cropHeight: Int ) = {
    val imageMat = byteArrayToMat(bytes)
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    val imageArray = matToArray(imageCent)
    val imageNorm = channelScaledNormalize(imageArray, 127, 127, 127, 1/127f)
    imageNorm
  }

  def processForTFNetResNet(bytes: Array[Byte], cropWidth: Int, cropHeight: Int) = {
    val imageMat = byteArrayToMat(bytes)
    val imageCent = centerCrop(imageMat, cropWidth, cropHeight)
    val imageArray = matToArray(imageCent)
    imageArray
  }

}

