package io.codifica.algorithms.data

import breeze.linalg.DenseMatrix

object TrainingAndTestSet {

  def apply(dataset: DenseMatrix[Double]) : TrainingAndTestSet = {
    val datasetWithOnesColumn = DenseMatrix.horzcat(DenseMatrix.ones[Double](dataset.rows, 1), dataset(0 until dataset.rows, 0 until dataset.cols))

    val trainingSetSize = (datasetWithOnesColumn.rows * 0.65).toInt
    val trainingSet = datasetWithOnesColumn(0 until trainingSetSize, 0 until datasetWithOnesColumn.cols)
    val testSet = datasetWithOnesColumn(trainingSetSize until datasetWithOnesColumn.rows, 0 until datasetWithOnesColumn.cols)

    new TrainingAndTestSet(Matrix(trainingSet), Matrix(testSet))
  }

}

class TrainingAndTestSet(val trainingSet: Matrix, val testSet: Matrix)
