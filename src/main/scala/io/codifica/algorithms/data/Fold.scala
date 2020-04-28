package io.codifica.algorithms.data

import breeze.linalg.DenseMatrix

object Fold {

  /**
   * The general procedure is as follows:
   *
   * Shuffle the dataset randomly.
   * Split the dataset into k groups
   * For each unique group:
   * Take the group as a hold out or test data set
   * Take the remaining groups as a training data set
   *
   * @param dataset
   * @param numberOfFolds
   * @return
   */
  def apply(dataset: Matrix, numberOfFolds: Int): List[Fold] = {
    val randomizedDataset = dataset.randomize
    val numberOfColumns = randomizedDataset.cols
    val itemsInFold = randomizedDataset.rows / numberOfFolds

    (for {
      i <- 0 until numberOfFolds
      from = i * itemsInFold
      to = if (from + itemsInFold > randomizedDataset.rows || i == numberOfFolds - 1) randomizedDataset.rows else from + itemsInFold

      crossValidationSet = randomizedDataset(from until to, 0 until numberOfColumns)
      trainingSet = DenseMatrix.vertcat(
        randomizedDataset(0 until from, 0 until numberOfColumns),
        randomizedDataset(to until randomizedDataset.rows, 0 until numberOfColumns))
    } yield new Fold(Matrix(trainingSet), Matrix(crossValidationSet)))
      .toList
  }

}

class Fold(val trainingSet: Matrix, val crossValidationSet: Matrix) {
  def prepare(): (FeaturesAndOutcomeSet, FeaturesAndOutcomeSet) = (FeaturesAndOutcomeSet(trainingSet), FeaturesAndOutcomeSet(crossValidationSet))
}

