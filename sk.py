from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# fs = np.load('angs10.npz')
# angs = fs['angles']
# pts = fs['pts']

class ArmCalibrate:
    def __init__(self):
        pass

    def fit(self, angs, pts):
        print(angs.shape)
        print(pts.shape)

        model1 = RANSACRegressor(LinearRegression())
        model2 = RANSACRegressor(LinearRegression())
        model1.fit(angs[:,[0]], pts[:,0])
        model2.fit(angs[:,[2]], pts[:,1])

        self.m1, self.b1 = float(model1.estimator_.coef_), model1.estimator_.intercept_
        self.m2, self.b2 = float(model2.estimator_.coef_), model2.estimator_.intercept_
        print('Coefficients :')
        print(self.m1, self.b1, self.m2, self.b2)

        # plt.scatter(angs[:,0], pts[:,0])
        # plt.scatter(angs[:,2], pts[:,1])
        # plt.plot([70, 100], [self.m1*70+self.b1, self.m1*100+self.b1])
        # plt.plot([5, 40], [self.m2*5+self.b2, self.m2*40+self.b2])
        # plt.show()

    def predict(self, ang):
        x, _, y = ang
        return (self.m1 * x + self.b1, self.m2 * y + self.b2)

    def inv_predict(self, pts):
        x, y = pts
        return ((x - self.b1) / self.m1, (y - self.b2) / self.m2)
