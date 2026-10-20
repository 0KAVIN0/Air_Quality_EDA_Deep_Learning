# Project License and Dataset Attribution

## Software License (MIT License)

Copyright (c) 2026 Kavin M (24AD053), KPR Institute of Engineering and Technology.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Dataset Attribution and Citation

### Dataset Name
**Air Quality Dataset** (UCI Machine Learning Repository)

### Persistent Identifier / DOI
- **DOI:** [https://doi.org/10.24432/C59K5F](https://doi.org/10.24432/C59K5F)
- **Repository URI:** [https://archive.ics.uci.edu/dataset/360/air%20quality](https://archive.ics.uci.edu/dataset/360/air%20quality)

### Dataset License
This dataset is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to share and adapt the material for any purpose, provided appropriate credit is given.

### Original Research Paper Citation
If using or referencing this dataset, please cite the original study:

```bibtex
@article{devito2008field,
  title={On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario},
  author={De Vito, Saverio and Massera, Ettore and Piga, Marco and Martinotto, Luca and Di Francia, Girolamo},
  journal={Sensors and Actuators B: Chemical},
  volume={129},
  number={2},
  pages={750--757},
  year={2008},
  publisher={Elsevier},
  doi={10.1016/j.snb.2007.09.060}
}
```

### Dataset Summary and Limitations
- **Data Period:** March 10, 2004 to April 4, 2005 (approximately one year of continuous hourly field recording).
- **Location:** Italian city, significantly polluted urban road environment.
- **Analyte Ground Truth:** Measured using certified co-located reference analyzers for CO, Non-Methane Hydrocarbons (NMHC), Benzene (C6H6), and Nitrogen Oxides (NOx/NO2).
- **Multisensor Array:** 5 solid-state metal oxide chemical sensors (PT08.S1 to PT08.S5).
- **Missing Value Sentinel:** `-200.0` represents sensor or reference analyzer dropouts and must be treated as missing data.
- **Reference Limitations:** `NMHC(GT)` was discontinued early in the recording period, resulting in over 90% missing values.
