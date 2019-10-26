# PreSense Visualizer Demo

This demo allows users to visualize data coming from their radar platforms using the OpenRadar DSP stack. The visualizers are organized into two categories.

1. Main Visualizer (main.py)
2. Realtime Visualizer (realtime.py)

## Description
### Main Visualizer
The Main Visualizer is to be used for visualizing data that has already been captured. The program will take a .bin file that has been post-processed (TI) and plot any of the four pre-defined plotting options below.
1. Range Doppler Plot
2. XY 2D Scatter (Range-Azimuth POV)
3. XZ 2D Scatter (Radar POV)
4. XYZ 3D Scatter

### Realtime Visualizer
The Realtime Visualizer is to be used for visualizing data that is streaming into the PC from the radar. The same four pre-defined plotting options are used.
1. Range Doppler Plot
2. XY 2D Scatter (Range-Azimuth POV)
3. XZ 2D Scatter (Radar POV)
4. XYZ 3D Scatter

## Usage
main.py is run as follows:
```
usage: main.py [-h] [--adc_data ADC_DATA] [--movie PLOT_MAKE_MOVIE]
               [--movie_dir MOVIE_DIR] [--movie_title MOVIE_TITLE]
               [--plot PLOT_CHOICE]
```

realtime.py is run as follows:
```
usage: realtime.py [-h] [--plot PLOT_CHOICE]
```

For help, run:
```
python main.py -h
```
```
python realtime.py -h
```

## Examples
```
main.py --adc_data ./data/1_person_walking_128loops.bin --plot 1
```

```
realtime.py --plot 1
```