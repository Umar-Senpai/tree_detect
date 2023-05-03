import numpy as np
from robot import robot
from RangeBearing import RangeBearing

class EKF():

    def __init__(self, STATE_SIZE, LM_SIZE, MAX_LANDMARKS, M_DIST_TH):
        self.STATE_SIZE = STATE_SIZE
        self.LM_SIZE = LM_SIZE
        self.NUM_LM = 0
        self.TOTAL_SIZE = STATE_SIZE + LM_SIZE*MAX_LANDMARKS
        self.X_ = np.zeros((self.TOTAL_SIZE, 1))
        self.P_ = np.zeros((self.TOTAL_SIZE, self.TOTAL_SIZE))
        self.M_DIST_TH = M_DIST_TH

        self.Q = None
        self.R = None

        self.ROBOT = robot()
        self.SENSOR = RangeBearing()

    def setQ(self, Q):
        '''
        Set covariance matrix for the motion model
        '''
        self.Q = Q

    def setR(self, R):
        '''
        Set covariance matrix for the measurement model
        '''
        self.R = R

    def X(self):
        '''
        return a numpy view of shape (3,3) representing the total state vector
        '''
        return self.X_

    def P(self):
        '''
        return a numpy view of shape (3,3) representing the total state covariance
        '''
        return self.P_
    
    def Xr(self):
        '''
        return a numpy view of shape (3,1) representing the robot state vector
        '''
        return self.X_[0:self.STATE_SIZE]
    
    def Prr(self):
        '''
        return a numpy view of shape (3,3) representing the robot state covariance
        '''
        return self.P_[0:self.STATE_SIZE, 0:self.STATE_SIZE]

    def xl(self, idx):
        '''
        return a numpy view of shape (s,) representing a landmark state (s is 2 for a pinger [x_,y])
        Input:
            - idx: an integer representing the landmark position in the map
        '''

        start = self.STATE_SIZE + idx * self.LM_SIZE
        end = start + self.LM_SIZE
        return self.X_[start:end, 0]

    def Prm(self):
        '''
        return a numpy view of shape (3, n * s) representing the robot-map correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[0:self.STATE_SIZE, self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM]

    def Pmr(self):
        '''
        return a numpy view of shape (n * s, 3) representing the map-robot correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, 0:self.STATE_SIZE]
        
    def Pmm(self):
        '''
        return a numpy view of shape (n * s, n * s) representing the map covariance matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[self.STATE_SIZE + self.LM_SIZE*self.NUM_LM:, self.STATE_SIZE + self.LM_SIZE*self.NUM_LM:]

    def Prl(self, idx):
        '''
        return a numpy view of shape (3, s) representing a robot-landmark correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        init = self.STATE_SIZE + self.LM_SIZE*idx
        end = init + self.LM_SIZE
        return self.P_[0:self.STATE_SIZE, init:end]

    def Plr(self, idx):
        '''
        return a numpy view of shape (s, 3) representing a landmark-robot correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        init = self.STATE_SIZE + self.LM_SIZE*idx
        end = init + self.LM_SIZE
        return self.P_[init:end, 0:self.STATE_SIZE]

    def Plm(self, idx):
        '''
        return a numpy view of shape (s, n * s) representing a landmark-mark correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        init = self.STATE_SIZE+self.LM_SIZE*idx
        end = init + self.LM_SIZE
        return self.P_[init:end, self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM]


    def Pml(self, idx):
        '''
        return a numpy view of shape (n * s, s) representing a map-landmark correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        init = self.STATE_SIZE+self.LM_SIZE*idx
        end = init + self.LM_SIZE
        return self.P_[self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, init:end]


    def Pll(self, idx):
        '''
        return a numpy view of shape (s,s) representing the landmark covariance matrix where s is the size of the landmark
        Input:
            -idx : an integer representing the landmark position in the map
        '''
        #
        init = self.STATE_SIZE+self.LM_SIZE*idx
        end = init + self.LM_SIZE
        return self.P_[init:end, init:end]


    def prediction(self, u, dt):
        '''
        Kalman Prediction step for the robot model
        u is a 2x1 np.array where u[0, 0] = v & w[1, 0] = w 
        '''
        self.X_[0:self.STATE_SIZE, 0] = self.ROBOT.fr(self.X_[0:self.STATE_SIZE, 0], u[0, 0], u[1, 0], dt)

        self.P_[0:self.STATE_SIZE, 0:self.STATE_SIZE] = self.ROBOT.Jfx( self.X_[0:self.STATE_SIZE, 0], u[0, 0], dt)@self.P_[0:self.STATE_SIZE, 0:self.STATE_SIZE]@self.ROBOT.Jfx( self.X_[0:self.STATE_SIZE, 0], u[0, 0], dt).T + self.ROBOT.Jfw(self.X_[0:self.STATE_SIZE, 0], dt)@self.Q@self.ROBOT.Jfw( self.X_[0:self.STATE_SIZE, 0], dt).T

        self.P_[0:self.STATE_SIZE, self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM] = self.ROBOT.Jfx(self.X_[0:self.STATE_SIZE, 0], u[0, 0], dt)@self.Prm()

        self.P_[self.STATE_SIZE:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, 0:self.STATE_SIZE] = self.Prm().T


    def update(self, y, idx):
        '''
        Kalman update step for the reange bearing system
        takes as input the measurement and the given landmark li
        '''
        z = y - self.SENSOR.h(self.X_[0:self.STATE_SIZE, 0], self.xl(idx))

        Hr = self.SENSOR.Jhxr(self.X_[0:self.STATE_SIZE, 0], self.xl(idx))

        Hli = self.SENSOR.Jhxl(self.X_[0:self.STATE_SIZE, 0], self.xl(idx))

        H = np.block([Hr, Hli])

        P_temp = np.block([[self.Prr(), self.Prl(idx)],
                      [self.Plr(idx), self.Pll(idx)]])

        Z = H@P_temp@H.T + self.R

        P_temp = np.block([[self.Prr(), self.Prl(idx)],
                           [self.Pmr(), self.Pml(idx)]])

        K = P_temp@H.T@np.linalg.inv(Z)

        self.X_[0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, [0]] = self.X_[0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, [0]] + K@z.reshape(2, 1)

        self.P_[0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, 0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM] = self.P_[0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM, 0:self.STATE_SIZE + self.LM_SIZE*self.NUM_LM] - K@Z@K.T

    def new_landmark(self, z):
        '''
        Method for adding a new landmark into the state vector
        '''
        L = self.SENSOR.g(self.X_[0:self.STATE_SIZE, 0], z)

        Gr = self.SENSOR.Jgxr(self.X_[0:self.STATE_SIZE, 0], z)

        Gy = self.SENSOR.Jgz(self.X_[0:self.STATE_SIZE, 0], z)

        Pll = Gr@self.Prr()@Gr.T + Gy@self.R@Gy.T
        Plx = Gr@np.block([self.Prr(), self.Prm()])

        a = self.STATE_SIZE + self.LM_SIZE*self.NUM_LM

        b = a + self.LM_SIZE

        self.X_[a:b, 0] = L

        self.P_[0:a, a:b] = Plx.T

        self.P_[a:b, 0:a] = Plx

        self.P_[a:b, a:b] = Pll

        self.NUM_LM += 1

    def data_association(self, y):
        '''
        Data association using the nearest neighbor data association algorithm 
        based on the mahalanobis distance
        '''
        mahalanobis = []

        for idx in range(self.NUM_LM):

            h =  self.SENSOR.h(self.X_[0:self.STATE_SIZE, 0], self.xl(idx))

            z = (y - h)
            
            if  np.abs(y[0] - h[0]) < 3 and np.abs(np.abs(y[1]) - np.abs(h[1])) < 1 and y[1]*h[1] < 0:
                z = np.array([y[0] - h[0], self.ROBOT.angle_wrap(np.abs(y[1]) - np.abs(h[1]))])

            z = z.reshape((2, 1))

            Jhx = self.SENSOR.Jhxr(self.X_[0:self.STATE_SIZE, 0], self.xl(idx))

            S = Jhx@self.Prr()@Jhx.T + self.R

            D = z.T@np.linalg.inv(S)@z

            mahalanobis.append(D[0, 0])

        mahalanobis.append(self.M_DIST_TH)

        return np.argmin(mahalanobis), np.min(mahalanobis)

        