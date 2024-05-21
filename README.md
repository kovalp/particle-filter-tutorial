# Installation
All code is written in Python. Poetry is used to track the package dependencies.

```commandline
poetry install
```





# Documentation

Details on the algorithms, background information in general and an documentation for all the main scripts can be found [in this paper](https://www.mdpi.com/1424-8220/21/2/438).

# Running the code

* ``demo_running_example``: runs the basic particle filter
* ``demo_range_only``: runs the basic particle filter with a lower number of landmarks (illustrates the particle filter's ability to represent non-Gaussian distributions).

```commandline
python src/particle_filter_tutorial/demos/demo_running_example/demo_running_example.py
```

Whenever running the code, a robot localization problem will be simulated. For most scripts, the visualization below should appear.

![alt text](https://github.com/jelfring/particle-filter-tutorial/blob/master/images/running_example_screenshot.png?raw=true)

The picture shows a top view of a 2D simulated world. Four landmarks can be observed by the robot (blue rectangles). The landmark positions are given in the map and therefore are used to estimate the tru robot position and orientation (red circle). The particles that together represent the posterior distribution are represented by the green dots.

Besides the standard particle filter, more advanced particle filters are implemented, different resampling schemes and different resampling algorithms are available. This allows for trying many different particle filter is similar settings.

The supported resampling algorithms are:
* Multinomial resampling
* Residual resampling
* Stratified resampling
* Systematic resampling

Supported resampling schemes are:
* Every time step
* Based on approximated effective number of particles
* Based on reciprocal of maximum particle weight

More advanced particle filters that are supported:
* Adaptive particle filter
* Auxiliary particle filter
* Extended Kalman particle filter

# Reference
In case you use code in your work, please cite:

Elfring J, Torta E, van de Molengraft R. Particle Filters: A Hands-On Tutorial. Sensors. 2021; 21(2):438.