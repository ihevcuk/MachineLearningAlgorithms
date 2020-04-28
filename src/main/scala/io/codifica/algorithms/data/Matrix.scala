package io.codifica.algorithms.data

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Random

object Matrix {

  def apply(xs: DenseMatrix[Double]) = new Matrix(xs)
  def apply(xs: DenseVector[Double]) = new Matrix(xs.toDenseMatrix.t)

}

class Matrix(m: DenseMatrix[Double]) {
  val matrix = m

  def printSize(text: String = "") = {
    println(s"$text ${size}")
  }

  def size(): (Int, Int) = {
    (matrix.rows, matrix.cols)
  }

  def randomize(): DenseMatrix[Double] = {
    val randomIndices = Random.shuffle(for(i <- 0 until matrix.rows) yield i)
    val randomized = matrix.copy

    for (i <- randomIndices.indices) {
      randomized(i, ::) := matrix(randomIndices(i), ::)
    }

    randomized
  }

  def print(title: String = ""): Unit = {
    println(s"$title ${size()}")
    for (row <- 1 to matrix.rows) {
      Console.print(s"$row: | ")
      for (column <- 1 to matrix.cols) {
        Console.print(s"${matrix.valueAt(row - 1, column - 1)}   ")
      }
      Console.print("|")
      println
    }
    println("------------------------------------")
  }

  def rows(): Int = matrix.rows

  def cols(): Int = matrix.cols

}
