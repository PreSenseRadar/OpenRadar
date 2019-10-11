# PreSense mmWave Package

This is PreSense team's implementation of TI mmwave radar DSP stack and some demos.
We are grateful for TI, TI's e2e forum and other people's help to make this happen.
Please star us if you like this repository and please consider citing this repo if you used it in your research.

The toolbox is modularized into separate steps
1. Reading raw ADC data.
2. Preprocessing data in DSP stack.
3. Utilizing preprocessed data for tracking, clustering and machine learning.
4. Different demo implementations from TI and our own explorations.

## Documentation
- [openradar.readthedocs.io](https://openradar.readthedocs.io)

## Contact 

- Please submit issues to our [GitHub](https://github.com/presenseradar/openradar) if you found any bugs or have any suggestions
- For anything else, send an email at presenseradar@gmail.com

## Directory Structure
    .
    ├── data                    # Small size sample data.
    ├── demo                    # Python implementations of TI demos.
    ├── docs                    # Documentation for mmwave pacakge and hardware setup.
    ├── mmwave                  # mmwave package including all the DSP, tracking, etc algorithms.
    ├── PreSense Applied radar  # Jupyter notebook series explaining how apply radar concepts to real data
    ├── scripts                 # Various setup scripts for mmwavestudio, etc
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── requirements.txt        # Required dependencies to run this package.
    └── setup.py                # Install mmwave package.

## Current Roadmap for this package
- [ ] Code refactoring for better API support.
- [ ] More tutorials to help people get started on FMCW radar.
- [ ] More AoA methods support.
- [ ] More noise removal algorithms.

## Future Plan
1. Hardware Abstraction Layer, e.g. hardware-agnostic data reading or processing.
2. ML on radar (classification, detection, tracking, etc).


## Installation

### Pip installation
```
Coming soon
```

### Debug Mode
```
git clone https://github.com/presenseradar/openradar
cd openradar
pip install -r requirements.txt
python setup.py develop
```

## Uninstallation

```
pip uninstall mmwave
```

## Example Import and Usage

```python
import mmwave as mm
from mmwave.dataloader import DCA1000

dca = DCA1000()
adc_data = dca.read()
radar_cube = mm.dsp.range_processing(adc_data)
```

## Acknowledgement
The PreSense team wants to thank Prof. [Sanjay Patel](https://ece.illinois.edu/directory/profile/sjp), without whom this project wouldn't be possible. We are also grateful for the generous sponsorship from The IBM-ILLINOIS Center for Cognitive Computing Systems Research ([C3SR](https://www.c3sr.com/)) and the guidance/leadership from [Jinjun Xiong](https://researcher.watson.ibm.com/researcher/view.php?person=us-jinjun) of IBM and Prof. [Wen-Mei Hwu](https://ece.illinois.edu/directory/profile/w-hwu). Special thanks to Prof. [Haitham Hassanieh](http://haitham.ece.illinois.edu/) and Prof. [Minh M. Do](http://minhdo.ece.illinois.edu/) for the techical advice and Texas Instruments&trade; for the hardware support.
