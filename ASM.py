# Implementing ASM algorithm
#
# references:
#
# Stegmann, Mikkel B, and David Delgado Gomez. n.d. “A Brief Introduction to Statistical Shape Analysis,” 15.
# Cootes, Tim. “An Introduction to Active Shape Models.” Image Processing and Analysis, January 1, 2000

import numpy as np
import matplotlib.pyplot as plt

import json
import random

decoder = json.decoder

def rotationMatrix(fi):
  return np.array([[np.cos(fi), -np.sin(fi)],[np.sin(fi), np.cos(fi)]])

def procrustesDistance(X1, X2):
  return np.sqrt(np.sum(np.square(X1 - X2)))

def centerOfGravity(X):
  M = X.shape[0]
  return np.sum(X, axis=0, keepdims = True)/M

def sizeFrobenious(X, centroidFun=centerOfGravity):
  center = centroidFun(X)
  return np.sqrt(np.sum(np.square(X-center)))

def centroidSize(X, centroidFun=centerOfGravity):
  center = centroidFun(X)
  return np.sum(np.sqrt(np.sum(np.square(X - center), axis = 1)), axis = 0)

def toPCAform(shapes):
  varShapes = []

  # reshape shapes to (x1, x2, ..., y1, y2, ...) to avoid tensor SVD when doing PCA
  for shape in shapes:
    varShapes.append(shape.T.reshape(-1,1))

  return varShapes


if __name__ == '__main__':
  # Choose size metric and centroid function
  sizeFun = sizeFrobenious
  centroidFun = centerOfGravity

  shapes = []
  tmpshapes = []

  with open('data/hand-landmarks.json', 'r') as saveFile:
    for line in saveFile.readlines():
      a = json.loads(line)
      shapes.append(np.array(a["coords"]))
      tmpshapes.append(None)

  # Shuffle data
  random.shuffle(shapes)

  # Take the first shape as mean, center it at origin
  mean = shapes[0]
  mean -= centroidFun(mean)
  mean /= sizeFun(mean)

  # Make mean hand shape look up
  mean = rotationMatrix(-np.pi/2).dot(mean.T).T

  # Center shapes at origin, resize them to unity
  for shape in shapes:
    # Center shapes around 0
    shape -= centroidFun(shape)

    # Resize to unity
    shape /= sizeFun(shape)


  # Calculate mean iteratively
  for iteration in range(10):
    tmpshapes = []

    for shape in shapes:
      # Rotate shape to mean
      # SVD
      corelationMatrix = mean.T.dot(shape)
      U, S, V = np.linalg.svd(corelationMatrix)

      # Rotate
      rotation = U.dot(V.T)
      tmpshapes.append(rotation.dot(shape.T).T)

    # Calculte mean, and normalize
    mean = sum(tmpshapes)/len(tmpshapes)
    mean /= sizeFun(mean)

  # Display mean shape
  plt.plot(*mean.T)
  plt.scatter(*centroidFun(mean).T)
  plt.show()
