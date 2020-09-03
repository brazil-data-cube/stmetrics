"""Unit-test for stmetrics."""


def test_getmetrics():
	import numpy
	series = numpy.array([0.157327502966,0.168894290924,0.141409546137,0.113800831139,0.0922891944647,0.0747280195355,0.0537555813789,0.0660935789347,0.0770644843578,0.0739007592201,0.0983928665519	,0.192401319742,0.286366194487,0.367539167404,0.420437157154,0.418041080236,0.413386583328	,0.375436246395,0.335108757019,0.307270467281,0.250428706408,0.178802281618,0.117247626185	,0.11457183212,0.103006377816,0.115561470389,0.114221975207,0.172464296222,0.284338653088	,0.386188000441,0.45704460144,0.571164608002,0.707974851131,0.648853778839,0.580699682236	,0.566288888454,0.547502994537,0.500209212303,0.447707682848,0.39193546772,0.357513874769	,0.290982276201,0.217830166221,0.148564651608,0.101060912013,0.111838668585	,0.121473513544	,0.113883294165,0.114351868629,0.116994164884,0.0982540994883,0.0843055993319,0.0827744230628	,0.0758764594793,0.0936531722546,0.0942907482386,0.172556817532])

	import stmetrics
	
	stmetrics.metrics.get_metrics(series)

	pass

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

	fractais = {'dfa_fd': 0.750251960291734,
			 'hurst_exp': -1.4554390466381768,
			 'katz_fd': 1.0606600552401722}

	bmetrics = stmetrics.fractal.ts_fractal(numpy.ones((1,360)).T)

	assert fractais == bmetrics

# def test_polares():
	
# 	import stmetrics
# 	import numpy

# 	polares = {'ecc_metric': 1.0,
# 				 'gyration_radius': 1.0,
# 				 'area_ts': 3.1414331587110302,
# 				 'polar_balance': 1.4686870114880517e-16,
# 				 'angle': 0.0,
# 				 'area_q1': 0.7853582896777579,
# 				 'area_q2': 0.785358289677758,
# 				 'area_q3': 0.7853582896777582,
# 				 'area_q4': 0.7853582896777582,
# 				 'fill_rate': 0.0,
# 				 'shape_index': 1.000025385558271,
# 				 'fill_rate2': 0.0048866915378894095,
# 				 'symmetry_ts': 0.0}

# 	bmetrics = stmetrics.polar.ts_polar(numpy.ones((1,360)).T)

# 	assert polares == bmetrics

"""Unit-test for stmetrics."""

def test_utils():
	import numpy
	import stmetrics
	
	series = numpy.array([0.157327502966,0.168894290924,0.141409546137,0.113800831139,0.0922891944647,0.0747280195355,0.0537555813789,0.0660935789347,0.0770644843578,0.0739007592201,0.0983928665519	,0.192401319742,0.286366194487,0.367539167404,0.420437157154,0.418041080236,0.413386583328	,0.375436246395,0.335108757019,0.307270467281,0.250428706408,0.178802281618,0.117247626185	,0.11457183212,0.103006377816,0.115561470389,0.114221975207,0.172464296222,0.284338653088	,0.386188000441,0.45704460144,0.571164608002,0.707974851131,0.648853778839,0.580699682236	,0.566288888454,0.547502994537,0.500209212303,0.447707682848,0.39193546772,0.357513874769	,0.290982276201,0.217830166221,0.148564651608,0.101060912013,0.111838668585	,0.121473513544	,0.113883294165,0.114351868629,0.116994164884,0.0982540994883,0.0843055993319,0.0827744230628	,0.0758764594793,0.0936531722546,0.0942907482386,0.172556817532])

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