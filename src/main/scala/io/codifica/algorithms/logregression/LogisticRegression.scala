package io.codifica.algorithms.logregression

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, pow, sigmoid}
import breeze.optimize.{DiffFunction, LBFGS}
import io.codifica.algorithms.data.{FeaturesAndOutcomeSet, Matrix}
import io.codifica.tennis.data.Matrix

class LogisticRegression() {

  def train(set: FeaturesAndOutcomeSet): Matrix = {
    val x = set.features
    val y = set.outcome
    val lambda = 1d
    val initialTheta = DenseVector.zeros[Double](x.cols())

    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(theta: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val costAndGradient = calculateCostWithRegularization(theta, x, y, lambda)
        costAndGradient
      }
    }

    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 250, m=7) // m is the memory. anywhere between 3 and 7 is fine. The larger m, the more memory is needed.
    val optimum = lbfgs.minimize(f, initialTheta)

    Matrix(optimum)
  }

  /**
   * H = sigmoid(X * theta);
   *
   * J = sum(-y' * log(H) - (1-y)' * log(1-H)) / m;
   * J = J + ((lambda/(2*m)) * sum(power(theta(2:size(theta,1)),2)));
   *
   * grad = (X' * (H - y)) / m;
   *
   * regMatrix = [0; theta(2:size(theta,1), :)] .* (lambda / m);
   *
   * grad = grad + regMatrix;
   *
   * @param theta parameters to train
   * @param features data to train on
   * @param outcome end result for each feature row
   * @param lambda gradient descent learning rate
   * @return cost and gradient
   */
  def calculateCostWithRegularization(theta: DenseVector[Double], features: Matrix, outcome: Matrix, lambda: Double) : (Double, DenseVector[Double]) = {
    val hypothesis = Matrix(sigmoid(features.matrix * theta))

    val cost: Double = calculateCost(hypothesis, features, outcome)
    val costRegularization: Double = calculateCostRegularization(theta, features, lambda)

    val gradient: DenseVector[Double] = calculateGradient(hypothesis, features, outcome)
    val gradientRegularization: DenseVector[Double] = calculateGradientRegularization(theta, features.rows(), lambda)

    (cost + costRegularization, gradient + gradientRegularization)
  }

  private def calculateCost(hypothesis: Matrix, features: Matrix, outcome: Matrix) : Double = {
    (-outcome.matrix.t * log(hypothesis.matrix) - (outcome.matrix * -1d + 1d).t * log(hypothesis.matrix * -1d + 1d)).valueAt(0, 0) / features.rows().toDouble
  }

  private def calculateCostRegularization(theta: DenseVector[Double], features: Matrix, lambda: Double) = {
    (lambda / (2 * features.rows())) * sum(pow(theta(1 until theta.length), 2))
  }

  private def calculateGradient(hypothesis: Matrix, features: Matrix, outcome: Matrix) : DenseVector[Double] = {
    val grad: DenseMatrix[Double] = (features.matrix.t * (hypothesis.matrix - outcome.matrix)) / hypothesis.rows().toDouble

    grad.toDenseVector
  }

  private def calculateGradientRegularization(theta: DenseVector[Double], setSize: Double, lambda: Double) = {
    val gradientTheta = theta.copy
    gradientTheta.update(0, 0d)

    gradientTheta * (lambda / setSize)
  }

  def predict(parameters: Matrix, set: Matrix) : Matrix = {
    Matrix(sigmoid(set.matrix * parameters.matrix).map(d => if (d > 0.5) 1d else 0d))
  }

}
