from enum import Enum

class Question(Enum):
    LinReg = "lg"
    FeatureScale = "fs"
    NormReg = "df"
    Gradient = "gd"
    K_Fold = "kf"