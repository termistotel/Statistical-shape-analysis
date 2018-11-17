# Implementing ASM algorithm with tensorflow for learning purposes
# This implementation is slightly slower and should not be used
#
# references:
#
# Stegmann, Mikkel B, and David Delgado Gomez. n.d. “A Brief Introduction to Statistical Shape Analysis,” 15.
# Cootes, Tim. “An Introduction to Active Shape Models.” Image Processing and Analysis, January 1, 2000

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import json
import random

decoder = json.decoder

def rotationMatrix(fi):
  return tf.constant(np.array([[np.cos(fi), -np.sin(fi)],[np.sin(fi), np.cos(fi)]]), name='rotation')

def procrustesDistance(X1, X2):
  return tf.sqrt(tf.reduce_sum(tf.square(X1, X2)))

def centerOfGravity(X):
  return tf.reduce_mean(X, axis=1, keepdims = True)

def sizeFrobenious(X, centroidFun=centerOfGravity):
  center = centroidFun(X)
  return tf.sqrt(tf.reduce_sum(tf.square(X-center)))

def centroidSize(X, centroidFun=centerOfGravity):
  center = centroidFun(X)
  return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(X - center), axis = 1)), axis = 0)

def tangentSpaceProjection(shapes, mean):
  meanSize = tf.reduce_sum(mean*mean)
  projections = []

  for shape in shapes:
    alpha = meanSize/tf.reduce_sum(mean*shape)
    projection = alpha * shape
    projections.append(projection)
  return projections


def toPCAform(shape):
  # reshape to (x1, x2, ..., y1, y2, ...) to avoid tensor SVD when doing PCA
  return tf.reshape(shape, (1, -1))

def toNormalForm(shape):
  # reshape to (m x 2), for m landmarks
  return tf.reshape(shape, (2, -1))




if __name__ == '__main__':
  # Choose size metric and centroid function
  sizeFun = sizeFrobenious
  centroidFun = centerOfGravity

  shapes = []
  tmpshapes = []

  # Load shapes
  with open('data/hand-landmarks.json', 'r') as saveFile:
    for line in saveFile.readlines():
      a = json.loads(line)
      shapes.append(np.array(a["coords"]).T)
      tmpshapes.append(None)

  # Shuffle data
  random.shuffle(shapes)

  # Initially, take the first shape as mean, center it at origin
  with tf.Session() as tmpsess:
    inmean = tf.constant(shapes[0])
    inmean = (inmean - centroidFun(inmean))/sizeFun(inmean)

    # Make mean hand shape look up
    inmean = tf.matmul(rotationMatrix(-np.pi/2), inmean)
    initialMean = tmpsess.run(inmean)

  # Define tensorflow graph
  graph1 = tf.Graph()
  with graph1.as_default():
    tfplaceholders = [tf.placeholder(dtype = tf.float64, shape = x.shape) for x in shapes]

    # Make mean a variable that will change iteratively
    mean = tf.Variable(initialMean)

    shapesAlligned = []
    for shape in tfplaceholders:
      # Center shapes at origin, resize them to unity
      normalized = (shape - centroidFun(shape))/sizeFun(shape)

      # Rotate shape to mean: SVD
      corelationMatrix = tf.matmul(mean, tf.matrix_transpose(normalized))
      S, U, V = tf.linalg.svd(corelationMatrix)
      rotation = tf.matmul(U, tf.matrix_transpose(V))

      # Rotate and save
      shapesAlligned.append(tf.matmul(rotation,normalized))

    # Update mean operation
    update = tf.assign(mean, sum(shapesAlligned)/len(shapesAlligned))



    # PCA

    # Ravel mean and shapes to vectors for PCA
    mean2 = toPCAform(mean)
    PCAshapes = map(toPCAform, shapesAlligned)

    # Project to tangent space
    PCAshapes = tangentSpaceProjection(PCAshapes, mean2)

    # prepare for covariance calculation
    N = tf.shape(mean2, out_type=tf.float64)[1]
    covarianceX = tf.zeros(dtype = tf.float64 , shape = (N, N))

    # Calculate shape covariance
    for shape in PCAshapes:
      diff = shape - mean2
      covarianceX += tf.matmul(tf.matrix_transpose(diff), diff)
    covarianceX /= N

    # Calculate eigenbasis and eigenvalue
    cov, U, Vt = tf.linalg.svd(covarianceX)

    # Standard deviations for each mode
    sigma = tf.reshape(tf.sqrt(cov), (-1, 1))

    # generating shape
    vec = tf.placeholder(dtype = tf.float64, shape = (None, None), name="vec")
    getShape = toNormalForm(mean2 + tf.matrix_transpose(tf.matmul(U, vec)))


  # Create feed dict for placeholders
  feed_dict = {}
  for i in range(len(tfplaceholders)):
    feed_dict[tfplaceholders[i]] = shapes[i]

  with tf.Session(graph=graph1) as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(10):
      sess.run(update, feed_dict=feed_dict)

    # Display mean shape
    plt.plot(*sess.run(mean))
    # for tfshape in shapesAlligned:
    #   plt.plot(*sess.run(tfshape, feed_dict=feed_dict))
    # plt.show()

    # Display a few shapes of +/- sigma for first mode
    s = sess.run(sigma, feed_dict=feed_dict)
    b = np.zeros(shape=s.shape)
    for i in range(9):
      b[0] = s[0]*0.25*i - s[0]
      feed_dict[vec] = b
      plt.plot(*sess.run(getShape, feed_dict=feed_dict))
    plt.show()

  # # Display a few shapes of +/- sigma for second mode
  # b = sigma*0
  # for i in range(9):
  #   b[1] = sigma[1]*0.25*i - sigma[1]
  #   x = mean2 + U.dot(b)
  #   plt.plot(*toNormalform(mean2).T)
  #   plt.plot(*toNormalform(x).T)
  #   plt.show()

