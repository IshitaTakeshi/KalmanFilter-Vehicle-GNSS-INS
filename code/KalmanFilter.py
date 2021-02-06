# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %Kalman Fliter Implementation on IMU+GNSS Recorded Data  %
# % by: Alireza Ahmadi                                     %
# % University of Bonn- MSc Robotics & Geodetic Engineering%
# % Alireza.Ahmadi@uni-bonn.de                             %
# % AhmadiAlireza.webs.com                                 %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%***************definition of parameters************************
#  a = 6378137.0;        %m Earth's ellipsoid radius at wquator
#  b = 6356752.3142 ;    %m Earth's ellipsoid radius at poles
#  ecc = 0.0818191908426;  %- Earth's ellipsoid eccentricity
#  w_i_e = 7.2921150*pow(10, -5);   %rad/s Earth's rotational speed
#  mu = 3.986004418*pow(10, 14);  %m^3/s^2 Geocentric gravitional constant
#  f = 1/298.257223563;  %- Earth's ellipsoid flattening
#  omega_ib_b = zeros(3,3);
#  g0 = 0;
#  R2D = 180/pi;
#  D2R = pi/180;
# %%***************************************************************

from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
np.set_printoptions(linewidth=120)


# filename0  = 'IMAR0000.mat';
# filename1  = 'IMAR0001.mat';
# filename2  = 'IMAR0002.mat';
filename3  = 'IMAR0003.mat';

# filename00  = 'UTM_IMAR0000.mat';
# filename01  = 'UTM_IMAR0001.mat';
# filename02  = 'UTM_IMAR0002.mat';
filename03  = 'UTM_IMAR0003.mat';

markersize = 0.1

imu = loadmat("IMAR0003.mat")["imu"]
imu_acc_ib_b = imu["acc_ib_b"][0][0]
imu_omg_ib_b = imu["omg_ib_b"][0][0]
imu_rpy_ned = imu["rpy_ned"][0][0]
imu_time = imu["imu_time"][0][0].flatten()

UTM = loadmat("UTM_IMAR0003.mat")["UTM"]

x_0 = UTM[0, 0];
y_0 = UTM[0, 1];
phi_0 = imu_rpy_ned[0, 2];
phi_D_0 = 0;
v_0 = 0;
a_0 = 0;

x_i = 0;
y_i = 0;
phi_i = 0;
phi_D_i = 0;
v_i = 0;
a_i = 0;

Delta_t = 0.001

Variance_phi_D = 0.0035;
Varince_a = pow(0.01, 2);

Variance_x = pow(10, -4);
Variance_y = pow(10, -4);
Variance_phi = pow(10, -5);

P_kM1 = np.array([
  [Variance_x, 0,          0,            0, 0, 0],
  [0,          Variance_y, 0,            0, 0, 0],
  [0,          0,          Variance_phi, 0, 0, 0],
  [0,          0,          0,            0, 0, 0],
  [0,          0,          0,            0, 0, 0],
  [0,          0,          0,            0, 0, 0]
])

cnt_b = 0;
Interval = 1;
P_Interval = 10;

plt.plot(UTM[:, 0], UTM[:, 1], 'o-',
         markersize=markersize,
         label="ground truth")

Ans = np.zeros((UTM.shape[0]-1, 2))

cnt=Interval;
while cnt < UTM.shape[0]:
    x_kM = np.array([
      x_0 + v_0 * Delta_t * np.cos(phi_0),
      y_0 + v_0 * Delta_t * np.sin(phi_0),
      phi_0 + phi_D_0 * Delta_t,
      phi_D_0,
      v_0 + a_0 * Delta_t,
      a_0
    ])

    Tans_Matrix = np.array([
      [1, 0, -v_0 * Delta_t * np.sin(phi_0), 0, Delta_t * np.cos(phi_0), 0],
      [0, 1,  v_0 * Delta_t * np.cos(phi_0), 0, Delta_t * np.sin(phi_0), 0],
      [0, 0,  1, Delta_t, 0, 0],
      [0, 0,  0, 1, 0, 0],
      [0, 0,  0, 0, 1, Delta_t],
      [0, 0,  0, 0, 0, 1]
    ])

    Q = np.array([
      [Variance_phi_D,         0],
      [             0, Varince_a]
    ])

    GQG_T = np.array([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, Variance_phi_D*Delta_t*Delta_t, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, Varince_a*Delta_t*Delta_t]
    ])
    P_k_M = Tans_Matrix.dot(P_kM1).dot(Tans_Matrix.T) + GQG_T;

    #  Varince_x_gps = pow(0.05, 2);
    #  Varince_y_gps = pow(0.05, 2);
    #  Variance_Acc  = pow(0.001, 2);
    #  Variance_gyro = 5*pow(10, -16);

    Varince_x_gps = 0.01;
    Varince_y_gps = 0.01;
    Variance_Acc  = 0.001;
    Variance_gyro = 5*pow(10, -5);

    z_k = np.array([
        UTM[cnt, 0],
        UTM[cnt, 1],
        -imu_acc_ib_b[cnt_b, 0],
         imu_omg_ib_b[cnt_b, 2]
    ])

    R_gps = np.array([
        [Varince_x_gps,             0,            0,             0],
        [0,             Varince_y_gps,            0,             0],
        [0,                         0, Variance_Acc,             0],
        [0,                         0,            0, Variance_gyro]
    ])

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0]
    ])

    h = np.dot(H, x_kM)

    S = H.dot(P_k_M).dot(H.T) + R_gps
    K_k = P_k_M.dot(H.T).dot(np.linalg.inv(S))

    x_k = x_kM + K_k.dot(z_k - h)

    P_k = (np.identity(6) - K_k.dot(H)).dot(P_k_M);

    Ans[cnt-1] = [x_k[0], x_k[1]]

    P_kM1 = P_k_M;

    x_0 = x_k[0];
    y_0 = x_k[1];
    phi_0 = x_k[2];
    phi_D_0 = x_k[3];
    v_0 = x_k[4];
    a_0 = x_k[5];

    cnt = cnt + Interval;
    cnt_b = cnt_b + P_Interval;

    Delta_t = (imu_time[cnt_b] - imu_time[cnt_b-P_Interval]);

print(Ans[:, 0])
print(Ans[:, 1])
plt.plot(Ans[:, 0], Ans[:, 1], 'o-',
        markersize=markersize, label="prediction")

plt.legend()

plt.show()
# xlabel('UTM-East');
# ylabel('UTM-North');
# legend([P1,P2],'UTM','EKF');
