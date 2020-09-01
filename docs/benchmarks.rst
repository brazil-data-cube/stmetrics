# Benchmarks

We assessed the performance of two main functions of stmetrics: `get_metrics` and `sits2metrics`. For that, we used a core i7-8700 CPU @ 3.2 GHz and 16GB of RAM memory. With this test we wanted to assess the performance of the package to compute the metrics available under different scenarios.

We compared the time and memory performance of those functions using different approaches. For `get_metrics` function, we assessed the performance using randomn time series, created with numpy, with different leghts. For the `sits2metrics` function we used images with different dimensions in columns and rows, maitaining the same length. 

# `get-metrics` analysis

To evaluate the performance of `get_metrics` function, we implemented a simple test using radomn time series built with `numpy` package, using the following code.


```python
import time
import stmetrics
import numpy
import matplotlib.pyplot as plt

tempos = []
for i in range(5,1000):
    start = time.time()
    stmetrics.metrics.get_metrics(numpy.random.rand(1,i)[0])
    end = time.time()
    tempos.append(end - start)

```

<p style="font-style: italics;" align="center">
<img height=384 src="figures/get_metrics.png" alt="Execution time of get_metrics functions with different time series lenghts." /><br>
Fig. 1: Executation time of get_metrics with time series with different lenghts.
</p>

The `get_metrics` function presents a linear response regarding the lenght of the time series, been able to compute the metrics for a time series with 1,000 data points in less than one second. Despite of that, for the following versions, we will try to improve this performance.

# `sits2metrics` analysis

<p style="font-style: italics;" align="center">
<img height=384 src="https://github.com/seung-lab/connected-components-3d/blob/master/benchmarks/cc3d_vs_scipy_single_label_10x.png" alt="Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (black) SciPy 1.3.0 (blue) cc3d 1.2.2" /><br>
Fig. 2: SciPy vs cc3d run ten times on a 512x512x512 connectomics segmentation masked to only contain one label. (black) SciPy 1.3.0 (blue) cc3d 1.2.2
</p> 
