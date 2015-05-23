BASE_INDEX = 1
"""Index of "base" points (96% of the training set picked at random)"""

VALID_INDEX = 2
"""Index of "valid" points (2% of the training set picked at random)"""

HIDDEN_INDEX = 3
"""Index of "hidden" points (2% of the training set picked at random)"""

PROBE_INDEX = 4
"""Index of "probe" points (1/3 of the test set picked at random)"""

QUAL_INDEX = 5
"""Index of "qual" points (2/3 of the test set picked at random)"""


USER_INDEX = 0
"""Index of user ID in data point tuple for all data point arrays"""

MOVIE_INDEX = 1
"""Index of movie ID in data point tuple for all data point arrays"""

TIME_INDEX = 2
"""Index of time stamp in data point tuple for all data point arrays"""

RATING_INDEX = 3
"""Index of rating in data point tuple for all data point arrays"""


SVD_FEATURE_VALUE_INITIAL = 0.01
"""Default value for initial algorithm predictions"""

BLENDING_RATIO = 25
"""Blending ratio (K) described by funny to blend global mean and movie mean"""
