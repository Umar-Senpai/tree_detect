import numpy as np

class robot():

    # The Differential Drive Prediction equations
    def fr(self, x_, v, w, dt) -> np.ndarray:
        x = x_[0] + np.cos(x_[2]) * v * dt
        y = x_[1] + np.sin(x_[2]) * v * dt
        theta = self.angle_wrap(x_[2] + w * dt)
        return np.array([x,y,theta]).T

    def Jfx(self, x_, v, dt) -> np.ndarray:
        return np.array([[1, 0, -(v * dt)*np.sin(x_[2])], 
                         [0, 1,  (v * dt)*np.cos(x_[2])], 
                         [0, 0,   1]])

    def Jfw(self,x_, dt = None) -> np.ndarray:
        return np.array([[np.cos(x_[2]) * dt, 0], 
                         [np.sin(x_[2]) * dt, 0], 
                         [0, dt]])

    def angle_wrap(self, ang):
        '''
        Return the angle normalized between [-pi, pi].

        Works with numbers and numpy arrays.

        :param ang: the input angle/s.
        :type ang: float, numpy.ndarray
        :returns: angle normalized between [-pi, pi].
        :rtype: float, numpy.ndarray
        '''
        return (ang + np.pi) % (2 * np.pi) - np.pi