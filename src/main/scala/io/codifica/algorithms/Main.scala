package io.codifica.algorithms

import breeze.linalg.sum
import breeze.numerics.abs
import io.codifica.algorithms.data.{FileLoader, Matrix}
import io.codifica.algorithms.logregression.LogisticRegression

object Main extends App {
  val sets = TrainingAndTestSet(FileLoader.load("features.csv"))
  val foldedSets = Fold(sets.trainingSet, 5)

  val learningAlgorithm = new LogisticRegression

  val accuracies =
    for {
      fold <- foldedSets
      modelTrainSets = fold.prepare()
      trainedParameters = learningAlgorithm.train(modelTrainSets._1)
      predictedOnTrainingSet = learningAlgorithm.predict(trainedParameters, modelTrainSets._1.features)
      accuracyOnTrainingSet = accuracy(predictedOnTrainingSet, modelTrainSets._1.outcome)
      predictedOnCrossValidationSet = learningAlgorithm.predict(trainedParameters, modelTrainSets._2.features)
      accuracyOnCrossValidationSet = accuracy(predictedOnCrossValidationSet, modelTrainSets._2.outcome)
    } yield (trainedParameters, accuracyOnTrainingSet, accuracyOnCrossValidationSet)

  val trainedParameters = accuracies.maxBy(_._3)._1

  val testSet = FeaturesAndOutcomeSet(sets.testSet)
  val predictedOnTestSet = learningAlgorithm.predict(trainedParameters, testSet.features)
  val accuracyOnTestSet = accuracy(predictedOnTestSet, testSet.outcome)

  def accuracy(predicted: Matrix, outcome: Matrix): Double = {
    val diff = predicted.matrix - outcome.matrix
    val wrong = sum(abs(diff))
    (predicted.rows - wrong) / predicted.rows.toDouble
  }

  accuracies.foreach(a => {
    println(s"Accuracy on training set = ${a._2}, accuracy on cross validation set = ${a._3}")
  })
  trainedParameters.print("Trained parameters")
  println(s"Accuracy on test set is $accuracyOnTestSet")

}
