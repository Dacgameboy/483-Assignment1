from lin_regr import *
from gradient import *

# Initialize the object
linRegr = LinRegr("Data1.csv", 5)
#gradient_descent = gd("Data1.csv")

myProblems = [Question.FeatureScale, Question.Gradient]

for Problem in myProblems:

    match Problem:

        case Question.Regular:

            linRegr.train(Problem)

        case Question.Gradient:

            gd = Gradient("Data1.csv", 0.000007, "Idx", 1, 5)
            gd.train()

        case Question.FeatureScale:

            linRegr.train(Problem)