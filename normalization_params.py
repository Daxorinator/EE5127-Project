# Normalization parameters for feature scaling
# Generated from UCI HAR dataset training data
# Units: accelerometer in g's, gyroscope in rad/s

NORMALIZATION_PARAMS = {
    'tBodyAcc-mean()-X': {'min': -0.26328375721874997, 'max': 0.14887772203125},
    'tBodyAcc-mean()-Y': {'min': -0.51552368046875, 'max': 0.5335020421875001},
    'tBodyAcc-mean()-Z': {'min': -0.2945620638046875, 'max': 0.36611885703125},
    'tBodyAcc-max()-X': {'min': -0.03268312, 'max': 1.299912},
    'tBodyAcc-max()-Y': {'min': -0.2556468, 'max': 0.9759764},
    'tBodyAcc-max()-Z': {'min': -0.09262088, 'max': 1.066916},
    'tBodyAcc-min()-X': {'min': -1.232238, 'max': 0.0931572},
    'tBodyAcc-min()-Y': {'min': -1.345267, 'max': 0.2357287},
    'tBodyAcc-min()-Z': {'min': -1.364707, 'max': 0.1033281},

    'tBodyAccJerk-mean()-X': {'min': -0.664351102362205, 'max': 0.5905879133858268},
    'tBodyAccJerk-mean()-Y': {'min': -0.42587149606299196, 'max': 0.37388566929133865},
    'tBodyAccJerk-mean()-Z': {'min': -0.4020321653543307, 'max': 0.36802818897637796},
    'tBodyAccJerk-max()-X': {'min': 0.12742757, 'max': 50.12208999999999},
    'tBodyAccJerk-max()-Y': {'min': 0.13674969, 'max': 40.250945},
    'tBodyAccJerk-max()-Z': {'min': 0.21645255000000002, 'max': 55.524165},
    'tBodyAccJerk-min()-X': {'min': -45.141071, 'max': -0.109377335},
    'tBodyAccJerk-min()-Y': {'min': -36.389145, 'max': -0.1361658},
    'tBodyAccJerk-min()-Z': {'min': -41.313185000000004, 'max': -0.21453255000000002},

    'tBodyGyro-mean()-X': {'min': -0.9141614453125, 'max': 0.7906608800468751},
    'tBodyGyro-mean()-Y': {'min': -0.3510968166015625, 'max': 0.485057559375},
    'tBodyGyro-mean()-Z': {'min': -0.4378068751484375, 'max': 0.40438009296875},
    'tBodyGyro-max()-X': {'min': -0.2417764, 'max': 4.155473},
    'tBodyGyro-max()-Y': {'min': -0.11351, 'max': 5.746062},
    'tBodyGyro-max()-Z': {'min': -0.2397656, 'max': 2.365982},
    'tBodyGyro-min()-X': {'min': -4.733656, 'max': 0.387086},
    'tBodyGyro-min()-Y': {'min': -5.97433, 'max': 0.2560976},
    'tBodyGyro-min()-Z': {'min': -2.763014, 'max': 0.2477599},

    'tBodyGyroJerk-mean()-X': {'min': -1.6592816141732283, 'max': 2.013248818897637},
    'tBodyGyroJerk-mean()-Y': {'min': -1.6457193700787398, 'max': 1.6168899606299212},
    'tBodyGyroJerk-mean()-Z': {'min': -1.0980803149606297, 'max': 1.1703807086614175},
    'tBodyGyroJerk-max()-X': {'min': 0.1614925, 'max': 100.94581},
    'tBodyGyroJerk-max()-Y': {'min': 0.181581265, 'max': 215.306335},
    'tBodyGyroJerk-max()-Z': {'min': 0.15593000000000135, 'max': 79.167395},
    'tBodyGyroJerk-min()-X': {'min': -112.5145, 'max': -0.1507297},
    'tBodyGyroJerk-min()-Y': {'min': -238.50659000000002, 'max': -0.18334175},
    'tBodyGyroJerk-min()-Z': {'min': -96.08938, 'max': -0.17956540000000001},

    'tBodyAccMag-mean()': {'min': 0.0042358725852490895, 'max': 0.685718928971917},
    'tBodyAccMag-max()': {'min': 0.008321346420900948, 'max': 1.8063824308000673},
    'tBodyAccMag-min()': {'min': 0.00010956880663646237, 'max': 0.363554788462686},

    'tBodyAccJerkMag-mean()': {'min': 0.1255243696702924, 'max': 18.126123530930997},
    'tBodyAccJerkMag-max()': {'min': 0.320158544455194, 'max': 60.90614806866397},
    'tBodyAccJerkMag-min()': {'min': 0.0027386966972403107, 'max': 3.448654463669273},
    
    'tBodyGyroMag-mean()': {'min': 0.004128067427194935, 'max': 1.9359035531458741},
    'tBodyGyroMag-max()': {'min': 0.010594028103109034, 'max': 6.042246673775695},
    'tBodyGyroMag-min()': {'min': 0.00024317714410044872, 'max': 0.5423309241385835},
    
    'tBodyGyroJerkMag-mean()': {'min': 0.1403115865422672, 'max': 62.07602056892374},
    'tBodyGyroJerkMag-max()': {'min': 0.3243723022092053, 'max': 242.35611226476342},
    'tBodyGyroJerkMag-min()': {'min': 0.0007116355317155123, 'max': 9.424757397117709},
    
    'fBodyAcc-mean()-X': {'min': 0.01237202527171472, 'max': 3.9550492481178683},
    'fBodyAcc-mean()-Y': {'min': 0.013518167394674752, 'max': 2.3190958596131033},
    'fBodyAcc-mean()-Z': {'min': 0.026219929696198213, 'max': 2.6359581036313107},
    'fBodyAcc-max()-X': {'min': 0.043341548266239774, 'max': 44.96954582227482},
    'fBodyAcc-max()-Y': {'min': 0.04932299113688046, 'max': 68.28826140000001},
    'fBodyAcc-max()-Z': {'min': 0.09054695355179632, 'max': 46.8632137},
    'fBodyAcc-min()-X': {'min': 3.803280920033326e-06, 'max': 0.49425421787520557},
    'fBodyAcc-min()-Y': {'min': 8.436517896746832e-06, 'max': 0.42677440846776277},
    'fBodyAcc-min()-Z': {'min': 1.783070220902935e-07, 'max': 0.5091322678922316},
    
    'fBodyAccJerk-mean()-X': {'min': 0.4700767095959454, 'max': 132.86966376822173},
    'fBodyAccJerk-mean()-Y': {'min': 0.46109863908701326, 'max': 76.81461601761482},
    'fBodyAccJerk-mean()-Z': {'min': 0.8851479224652207, 'max': 107.08668946483085},
    'fBodyAccJerk-max()-X': {'min': 1.335856096602167, 'max': 686.8009029215207},
    'fBodyAccJerk-max()-Y': {'min': 1.4134065132783364, 'max': 457.5129638665394},
    'fBodyAccJerk-max()-Z': {'min': 2.248117591917168, 'max': 559.5396843602708},
    'fBodyAccJerk-min()-X': {'min': 7.16500000006795e-05, 'max': 21.88960884398186},
    'fBodyAccJerk-min()-Y': {'min': 5.449999999962429e-06, 'max': 16.102625680765517},
    'fBodyAccJerk-min()-Z': {'min': 0.00018600000001978145, 'max': 15.340776451655227},
    
    'fBodyGyro-mean()-X': {'min': 0.016314320626294655, 'max': 9.422155819754991},
    'fBodyGyro-mean()-Y': {'min': 0.017308432722376386, 'max': 11.00466097339114},
    'fBodyGyro-mean()-Z': {'min': 0.01840706452738092, 'max': 5.828373414786849},
    'fBodyGyro-max()-X': {'min': 0.06607816321, 'max': 117.012665},
    'fBodyGyro-max()-Y': {'min': 0.0533851005760631, 'max': 92.28196159099338},
    'fBodyGyro-max()-Z': {'min': 0.06651151546447952, 'max': 76.96565781747961},
    'fBodyGyro-min()-X': {'min': 4.827285153027747e-06, 'max': 1.5661257894586074},
    'fBodyGyro-min()-Y': {'min': 1.234696198913497e-05, 'max': 1.3141493927496413},
    'fBodyGyro-min()-Z': {'min': 7.854520340315657e-06, 'max': 1.0660006431405673},
    
    'fBodyAccMag-mean()': {'min': 0.026807496997871966, 'max': 3.9551395414163233},
    'fBodyAccMag-max()': {'min': 0.5421916909118834, 'max': 87.77202290840538},
    'fBodyAccMag-min()': {'min': 3.1388124252400386e-05, 'max': 0.5184860239484556},

    'fBodyBodyAccJerkMag-mean()': {'min': 0.8684362800163632, 'max': 130.77585037845893},
    'fBodyBodyAccJerkMag-max()': {'min': 15.94159494812714, 'max': 2302.017688428236},
    'fBodyBodyAccJerkMag-min()': {'min': 0.003360149394130563, 'max': 21.49393752094221},

    'fBodyBodyGyroMag-mean()': {'min': 0.026278419794193035, 'max': 11.456083974120055},
    'fBodyBodyGyroMag-max()': {'min': 0.5283926306809517, 'max': 247.79565480267192},
    'fBodyBodyGyroMag-min()': {'min': 5.3471625088867444e-05, 'max': 1.2553463754790892},

    'fBodyBodyGyroJerkMag-mean()': {'min': 0.9301274404378079, 'max': 495.8660297608792},
    'fBodyBodyGyroJerkMag-max()': {'min': 17.819571490867936, 'max': 7883.654612253315},
    'fBodyBodyGyroJerkMag-min()': {'min': 0.0021022455958013436, 'max': 54.085061297994905},
}

def normalize_feature(value, feature_name):
    """Normalize a feature value to [-1, 1] range
    
    Args:
        value: Raw feature value (in g's for acceleration, rad/s for gyroscope)
        feature_name: Name of the feature (e.g., 'tBodyAcc-mean()-X')
    
    Returns:
        Normalized value in range [-1, 1]
    """
    params = NORMALIZATION_PARAMS.get(feature_name)
    if params is None:
        raise ValueError(f'Feature {feature_name} not found in normalization parameters')
    
    min_val = params['min']
    max_val = params['max']
    
    # Avoid division by zero
    if abs(max_val - min_val) < 1e-10:
        return 0.0
    
    # Normalize to [-1, 1]
    return 2 * (value - min_val) / (max_val - min_val) - 1
