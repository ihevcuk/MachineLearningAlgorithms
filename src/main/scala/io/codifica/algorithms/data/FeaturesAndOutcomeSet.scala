package io.codifica.algorithms.data

object FeaturesAndOutcomeSet {
  def apply(dataset: Matrix): FeaturesAndOutcomeSet = {
    val features = dataset.matrix(0 until dataset.rows, 0 until dataset.cols - 1)
    val outcome = dataset.matrix(::, dataset.cols - 1)

    new FeaturesAndOutcomeSet(Matrix(features), Matrix(outcome))
  }
}


class FeaturesAndOutcomeSet(val features: Matrix, val outcome: Matrix)
