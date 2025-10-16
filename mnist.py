import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
import matplotlib as plt