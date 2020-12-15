"""Unit-test for stmetrics."""


def test_getmetrics():
	import numpy
	import stmetrics
	
	series = numpy.array([0.157327502966,0.168894290924,0.141409546137,
		                  0.113800831139,0.0922891944647,0.0747280195355,
		                  0.0537555813789,0.0660935789347,0.0770644843578,
		                  0.0739007592201,0.0983928665519,0.192401319742,
		                  0.286366194487,0.367539167404,0.420437157154,
		                  0.418041080236,0.413386583328,0.375436246395,
		                  0.335108757019,0.307270467281,0.250428706408,
		                  0,1,0,
		                  0.103006377816,0.115561470389,0.114221975207,
		                  0.172464296222,0.284338653088,0.386188000441,
		                  0.45704460144,0.571164608002,0.707974851131,
		                  0.648853778839,0.580699682236,0.566288888454,
		                  0.547502994537,0.500209212303,0.447707682848,
		                  0.39193546772,0.357513874769,0.290982276201,
		                  0.217830166221,0.148564651608,0.101060912013,
		                  0.111838668585,0.121473513544,0.113883294165,
		                  0.114351868629,0.116994164884,0.0982540994883,
		                  0.0843055993319,0.0827744230628,0.0758764594793,
		                  0.0936531722546,0.0942907482386,0.172556817532])

	metrics = {'basics': {'max_ts': 0.707974,
						  'min_ts': 0.0,
						  'mean_ts': 0.237823,
						  'std_ts': 0.183005,
						  'sum_ts': 13.318112,
						  'amplitude_ts': 0.707974,
						  'mse_ts': 5.042865,
						  'fslope_ts': 0.250428,
						  'skew_ts': 0.795801,
						  'amd_ts': 0.043546,
						  'abs_sum_ts': 13.318112,
						  'iqr_ts': 0.28086,
						  'fqr_ts': 0.096272,
						  'tqr_ts': 0.380812,
						  'sqr_ts': 0.158729},
						 'polar': {'ecc_metric': 0.987689,
						  'gyration_radius': 0.378319,
						  'area_ts': 0.276252,
						  'polar_balance': 0.069048,
						  'angle': 3.541431,
						  'area_q1': 0.046879,
						  'area_q2': 0.033173,
						  'area_q3': 0.186429,
						  'area_q4': 0.00977,
						  'csi': 2.658336},
						 'fractal': {'dfa_fd': 2.053765, 'hurst_exp': 0.87168, 'katz_fd': 1.437053}}

	out = stmetrics.metrics.get_metrics(series,nodata=0.157327502966)
	assert metrics == out


def test_basics():

	import stmetrics
	import numpy

	basicas = {'max_ts': 1.0,
              'min_ts': 1.0,
              'mean_ts': 1.0,
              'std_ts': 0.0,
              'sum_ts': 360.0,
              'amplitude_ts': 0.0,
              'mse_ts': 360.0,
              'fslope_ts': 0.0,
              'skew_ts': 0.0,
              'amd_ts': 0.0,
              'abs_sum_ts': 360.0,
              'iqr_ts': 0.0,
              'fqr_ts': 1.0,
              'tqr_ts': 1.0,
              'sqr_ts': 1.0}

	bmetrics = stmetrics.basics.ts_basics(numpy.ones((1,360)).T)

	assert basicas == bmetrics

def test_fractal():
	
	import stmetrics
	import numpy

	fractais = {'dfa_fd': nan, 
                'hurst_exp': nan, 
                'katz_fd': nan}

	bmetrics = stmetrics.fractal.ts_fractal(numpy.ones((1,360)).T)

	assert fractais == bmetrics


def test_polares():
	
 	import stmetrics
 	import numpy

 	polares =  {'ecc_metric': 1.0,
              'gyration_radius': 1.0,
              'area_ts': 3.141433,
              'polar_balance': 0.0,
              'angle': 0.0,
              'area_q1': 0.785358,
              'area_q2': 0.785358,
              'area_q3': 0.785358,
              'area_q4': 0.785358,
              'csi': 1.000025}

 	bmetrics = stmetrics.polar.ts_polar(numpy.ones((360)))

 	assert polares == bmetrics


def test_utils():
	import numpy
	import stmetrics
	
	series = numpy.array([0.157327502966,0.168894290924,0.141409546137,
		                  0.113800831139,0.0922891944647,0.0747280195355,
		                  0.0537555813789,0.0660935789347,0.0770644843578,
		                  0.0739007592201,0.0983928665519,0.192401319742,
		                  0.286366194487,0.367539167404,0.420437157154,
		                  0.418041080236,0.413386583328,0.375436246395,
		                  0.335108757019,0.307270467281,0.250428706408,
		                  0.178802281618,0.117247626185,0.11457183212,
		                  0.103006377816,0.115561470389,0.114221975207,
		                  0.172464296222,0.284338653088,0.386188000441,
		                  0.45704460144,0.571164608002,0.707974851131,
		                  0.648853778839,0.580699682236,0.566288888454,
		                  0.547502994537,0.500209212303,0.447707682848,
		                  0.39193546772,0.357513874769,0.290982276201,
		                  0.217830166221,0.148564651608,0.101060912013,
		                  0.111838668585,0.121473513544,0.113883294165,
		                  0.114351868629,0.116994164884,0.0982540994883,
		                  0.0843055993319,0.0827744230628,0.0758764594793,
		                  0.0936531722546,0.0942907482386,0.172556817532])	


	geometry = stmetrics.utils.create_polygon(series)

	if geometry.is_valid == True:
		pass


def test_polar():
	import numpy
	from stmetrics import utils

	polares = utils.error_polar()

	if all(numpy.isnan(value) != numpy.nan for value in polares.values()) == True:
		pass


def test_fractal():
	import numpy
	from stmetrics import utils
	fractal = utils.error_fractal()

	if all(numpy.isnan(value) != numpy.nan for value in fractal.values()) == True:
		pass


def test_basics():
	import numpy
	from stmetrics import utils
	basics = utils.error_basics()

	if all(numpy.isnan(value) != numpy.nan for value in basics.values()) == True:
		pass

def test_symmetric_distance():
	import numpy
	from stmetrics import polar

	s1 = numpy.ones((360))
	s2 = numpy.ones((360))

	dist = polar.symmetric_distance(s1, s2)

	assert dist == 0

def test_symmetric_distance_ii():
	import numpy
	from stmetrics import polar

	s1 = numpy.ones((360))-0.1
	s2 = numpy.ones((360))

	dist = polar.symmetric_distance(s1, s2)

	assert dist == 0.596872

def test_geometries():

	from shapely import geometry
	from stmetrics import spatial

	p1 = geometry.Point(0,0)
	p2 = geometry.Point(1,0)
	p3 = geometry.Point(1,1)
	p4 = geometry.Point(0,1)

	pointList = [p1, p2, p3, p4, p1]

	poly = geometry.Polygon([[p.x, p.y] for p in pointList])

	out = [0.0, 1.0, 0.6376435773361453, 0.0, 1.0, 1.0]

	res = [spatial.symmetry(poly),
		   spatial.aspect_ratio(poly),
	 	   spatial.reock_compactness(poly),
	 	   spatial.rectangular_fit(poly),
	 	   spatial.width(poly),
		   spatial.length(poly)]

	assert out == res


def test_getmetrics_sits():
    import numpy
    from stmetrics import metrics

    out = numpy.array([  1.      ,   1.      ,   1.      ,   0.      , 360.      ,
				         0.      , 360.      ,   0.      ,   0.      ,   0.      ,
				       360.      ,   0.      ,   1.      ,   1.      ,   1.      ,
				         3.141433,   0.      ,   0.785358,   0.785358,   0.785358,   
				         0.785358,   0.      ,   1.      ,   1.      ,
				         1.000025])

    res = metrics._getmetrics(numpy.ones((360)))
    res = res[~numpy.isnan(res)]

    assert all(out == res)

def test_list_metrics():
	from stmetrics import utils

	out = ['max_ts',
		 'min_ts',
		 'mean_ts',
		 'std_ts',
		 'sum_ts',
		 'amplitude_ts',
		 'mse_ts',
		 'fslope_ts',
		 'skew_ts',
		 'amd_ts',
		 'abs_sum_ts',
		 'iqr_ts',
		 'fqr_ts',
		 'sqr_ts',
		 'tqr_ts',
		 'area_ts',
		 'angle',
		 'area_q1',
		 'area_q2',
		 'area_q3',
		 'area_q4',
		 'polar_balance',
		 'ecc_metric',
		 'gyration_radius',
		 'csi',
		 'dfa_fd',
		 'hurst_exp',
		 'katz_fd']

	assert all([out == utils.list_metrics()])


def test_sits2metrics_exception():
	import numpy
	import stmetrics
	import pytest

	with pytest.raises(Exception):
		assert stmetrics.metrics.sits2metrics([10])


def test_create_polygon_exception():
	import stmetrics
	import pytest

	with pytest.raises(Exception):
		assert stmetrics.utils.create_polygon([10])


def test_check_input_exception():
	import stmetrics
	import pytest

	with pytest.raises(Exception):
		assert stmetrics.utils.check_input([10])

 
def test_sits2metrics():

	import numpy
	import stmetrics

	sits = numpy.array([[[0.08213558, 0.58803765],
				        [0.49712389, 0.83526625]],

				       [[0.88548059, 0.30089922],
				        [0.46782818, 0.84561955]],

				       [[0.97508056, 0.37090787],
				        [0.23905704, 0.96134861]],

				       [[0.34126892, 0.0517639 ],
				        [0.56801062, 0.9046814 ]],

				       [[0.89621465, 0.79039706],
				        [0.76447722, 0.37223732]],

				       [[0.01181458, 0.92984248],
				        [0.95011783, 0.94595306]],

				       [[0.19884843, 0.86591456],
				        [0.25220217, 0.54905   ]],

				       [[0.44872961, 0.61002462],
				        [0.43320113, 0.41983541]],

				       [[0.67116755, 0.70299412],
				        [0.06319867, 0.99832697]],

				       [[0.57694712, 0.30948048],
				        [0.9029195 , 0.99803176]]])

	output = numpy.array([[[ 9.750800e-01,  9.298420e-01],
					       [ 9.501170e-01,  9.983260e-01]],

					      [[ 1.181400e-02,  5.176300e-02],
					       [ 6.319800e-02,  3.722370e-01]],

					      [[ 5.087680e-01,  5.520260e-01],
					       [ 5.138130e-01,  7.830350e-01]],

					      [[ 3.312370e-01,  2.702780e-01],
					       [ 2.762990e-01,  2.297330e-01]],

					      [[ 5.087687e+00,  5.520261e+00],
					       [ 5.138136e+00,  7.830350e+00]],

					      [[ 9.632650e-01,  8.780780e-01],
					       [ 8.869190e-01,  6.260890e-01]],

					      [[ 3.685641e+00,  3.777832e+00],
					       [ 3.403455e+00,  6.659211e+00]],

					      [[ 8.844000e-01,  7.386330e-01],
					       [ 8.397200e-01,  5.784910e-01]],

					      [[-4.801000e-02, -2.996340e-01],
					       [ 1.285020e-01, -8.078330e-01]],

					      [[ 4.132970e-01,  2.622960e-01],
					       [ 3.397510e-01,  2.659790e-01]],

					      [[ 5.087687e+00,  5.520261e+00],
					       [ 5.138136e+00,  7.830350e+00]],

					      [[ 5.974480e-01,  4.437080e-01],
					       [ 4.179080e-01,  3.368950e-01]],

					      [[ 2.700580e-01,  3.401940e-01],
					       [ 3.427010e-01,  6.921580e-01]],

					      [[ 5.128380e-01,  5.990310e-01],
					       [ 4.824760e-01,  8.751500e-01]],

					      [[ 7.783240e-01,  7.466950e-01],
					       [ 6.662430e-01,  9.536500e-01]],

					      [[ 7.090790e-01,  9.537960e-01],
					       [ 7.414750e-01,  1.785940e+00]],

					      [[ 1.396263e+00,  3.490658e+00],
					       [ 3.490658e+00,  5.585053e+00]],

					      [[ 1.183530e-01,  2.287100e-01],
					       [ 3.691700e-01,  3.263750e-01]],

					      [[ 3.475680e-01,  8.975300e-02],
					       [ 1.130390e-01,  5.781790e-01]],

					      [[ 6.238000e-02,  4.504290e-01],
					       [ 1.095530e-01,  2.568520e-01]],

					      [[ 1.807770e-01,  1.849030e-01],
					       [ 1.497120e-01,  6.245320e-01]],

					      [[ 1.068700e-01,  1.322950e-01],
					       [ 1.072770e-01,  1.576630e-01]],

					      [[ 9.558100e-01,  7.489210e-01],
					       [ 7.805490e-01,  9.796340e-01]],

					      [[ 1.767650e-01,  8.556510e-01],
					       [ 6.298040e-01,  6.739810e-01]],

					      [[ 3.372378e+00,  1.761302e+00],
					       [ 2.543674e+00,  1.508493e+00]],

					      [[-4.280310e-01, -7.486000e-03],
					       [ 1.888369e+00, -4.988190e-01]],

					      [[ 4.724270e-01,  4.724270e-01],
					       [ 6.548670e-01,  6.492550e-01]],

					      [[ 2.596693e+00,  1.818503e+00],
					       [ 2.289875e+00,  2.566626e+00]]])

	res = stmetrics.metrics.sits2metrics(sits)

	r1 = res.reshape(res.shape[0]*res.shape[1]*res.shape[2])
	r2 = output.reshape(output.shape[0]*output.shape[1]*output.shape[2])

	assert all(r1 == r2)


if __name__ == '__main__':
    pytest.main(['--color=auto', '--no-cov'])