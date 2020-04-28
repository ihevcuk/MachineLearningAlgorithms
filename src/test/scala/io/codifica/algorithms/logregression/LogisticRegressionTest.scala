package io.codifica.algorithms.logregression

import breeze.linalg.{DenseMatrix, DenseVector}
import io.codifica.algorithms.data.Matrix
import org.junit.Test
import org.junit.Assert._

class LogisticRegressionTest {

  @Test
  def costWithRegularizationTest() = {
    val x: DenseMatrix[Double] = DenseMatrix(
      (1.0d, 0.7d, 0.14d),
      (1.0d, 0.64d, 0.32d))

    val y: DenseMatrix[Double] = DenseMatrix(
      (1.0d),
      (0d))

    val theta: DenseMatrix[Double] = DenseMatrix(
      (2d),
      (3d),
      (5d))

    val alg = new LogisticRegression

    val cost = alg.calculateCostWithRegularization(theta.toDenseVector, Matrix(x), Matrix(y), 1d)

    assertEquals(cost._1, 11.26609, 0.00001)
    assertEquals(cost._2, DenseVector(0.49392378184892305, 1.8158663432487159, 2.65879024159544))
    println(cost)
  }


}
