package io.codifica.algorithms.data

import breeze.linalg.DenseMatrix

import scala.io.{BufferedSource, Source}

object FileLoader {

  def load(file: String) : DenseMatrix[Double] = {
    val bufferedSource : BufferedSource = Source.fromResource(file)

    val features: Iterator[Array[Double]] = for {
      line <- bufferedSource.getLines()
      cols = line.split(",").map(_.trim).map(_.toDouble)
    } yield cols

    val featuresArray = features.toArray

    bufferedSource.close

    // hack to get rows and columns like in csv
    new DenseMatrix(featuresArray(0).length, featuresArray.flatten, 0).t
  }

}
