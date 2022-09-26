from lin_regr import *
from gradient import *
from lasso_ridge_elastic import *

myProblems = [Question.K_Fold, Question.FeatureScale, Question.Gradient, Question.NormReg]

# myProblems = [Question.Gradient]

for Problem in myProblems:

    match Problem:

        case Question.LinReg:
            
            linRegr = LinRegr("Data1.csv", 5)
            linRegr.train(Problem)

        case Question.Gradient:

            gd = Gradient("Data1.csv", 0.000007, "Idx", 1, 5)
            gd.train()


        case Question.FeatureScale:

            linRegr = LinRegr("Data1.csv", 5)
            linRegr.train(Problem)

        case Question.NormReg:

            lre = LassoRidgeElastic("Data1.csv", 2)
            lre.train(Problem)
            
        case Question.K_Fold:

            lre = LassoRidgeElastic("Data1.csv", 2, 4)
            lre.train(Problem)