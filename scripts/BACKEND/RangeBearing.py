import numpy as np
import math

class RangeBearing:
    '''
    Class to encapsulate information provided by a range&bearing sensor that can detect pingers located in the world.
    The sensor is assumed to be located at the center of the robot.
    It stores:
    - time : float representing the timestamp of the reading
    - z    : a numpy vector of dimension (2,) representing the measured distance and angle
    - R    : a numpy matrix of dimension (2,2) representing the covariance matrix of the measurement
    - id   : int representing a unique identifier of the pinger (it solves the association problem)
    '''


    #def expected_measurement(self, x, mfeat):
    def h(self, xr, xl):
        '''
        Compute the expected measurement z = h(xr, xl, v), where xr is the robot state [x,y,\theta] and xl the pinger state [x,y]
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        Return: numpy array of shape (2,) representing the expected measurement [dist, angle]
        '''
        expected_distance = np.linalg.norm(xl - xr[0:2])
        
        x_r = xr[0]; y_r = xr[1];theta = self.angle_wrap(xr[2]); x_l = xl[0]; y_l = xl[1];
        x = (y_l - y_r)*np.sin(theta) + (x_l - x_r)*np.cos(theta) 
        y = (y_l - y_r)*np.cos(theta) - (x_l - x_r)*np.sin(theta)

        expected_angle = self.angle_wrap(np.arctan2(y, x))
        
        return np.array([expected_distance, expected_angle])


    def Jhxr(self, xr, xl) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to xr, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        x_l = xl[0]; y_l = xl[1]
        x_r = xr[0]; y_r = xr[1]

        m11 = (-x_l + x_r) / np.sqrt(x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2); m12 = (-y_l + y_r) / np.sqrt(x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2); m13 = 0
        m21 = (y_l - y_r) / (x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2);         m22 = (-x_l + x_r) / (x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2);        m23 = -1;
        return np.array([[m11, m12, m13], [m21, m22, m23]])

    def Jhxl(self, xr, xl) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to xl, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 2) (The Jacobian)
        '''
        x_l = xl[0]; y_l = xl[1]
        x_r = xr[0]; y_r = xr[1]

        m11 = (x_l - x_r) / np.sqrt(x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2); m12 = (y_l - y_r) / np.sqrt(x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2)
        m21 = (-y_l + y_r) / (y_l**2 - 2*y_l*y_r + y_r**2 + (x_l - x_r)**2);       m22 = (x_l - x_r) / (x_l**2 - 2*x_l*x_r + x_r**2 + (y_l - y_r)**2)

        return np.array([[m11, m12], [m21, m22]])
    
    def Jhv(xr = None, xl = None) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to v, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 2) (The Jacobian)
        '''
        # Noise is assumed independent (z = h(x) + v)
        return np.eye(2)

    def g(self, xr, z):
        '''
        Compute the inverse observation xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in z) 
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        Return: numpy array of shape (2,) representing the landmark expected position [x, y]
        '''
 
        theta_alpha = self.angle_wrap(xr[2] + z[1])
        x_l = xr[0] + z[0]*np.cos(theta_alpha)
        y_l = xr[1] + z[0]*np.sin(theta_alpha)
        return np.array([x_l, y_l])

    def Jgxr(self, xr, z) -> np.ndarray:
        '''
        Compute the Jacobian of xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in z) with respect to xr at point (xr, z)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        theta_alpha = self.angle_wrap(xr[2] + z[1])
        return np.array([[1, 0, -z[0]*np.sin(theta_alpha)], [0, 1, z[0]*np.cos(theta_alpha)]])

    def Jgz(self, xr, z) -> np.ndarray:
        '''
        Compute the Jacobian of xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in z) with respect to z at point (xr, z)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        theta_alpha = self.angle_wrap(xr[2] + z[1])
        m11 = np.cos(theta_alpha) ; m12 = -z[0]*np.sin(theta_alpha)
        m21 = np.sin(theta_alpha) ; m22 =  z[0]*np.cos(theta_alpha)
        
        return np.array([[m11, m12], [m21, m22]])

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