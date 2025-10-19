# Deep Learning Data Process Project

This is a data preprocessing utility library for deep learning tasks. This project provides various data processing functions and includes a dedicated testbench file to ensure all functionalities work correctly.

## Project Introduction

Data preprocessing is a crucial step in any deep learning workflow. This project aims to provide a set of reusable and easily testable scripts for processing and preparing datasets before feeding them into a model for training and evaluation.

## File Structure

This project mainly consists of the following two core files:

* `data_process.py`:
    * **Function**: Contains all the core data preprocessing functions.
    * **Examples **:
        * Normalization
        * Data Augmentation (e.g., random flips, crops)
        * Type Conversion (e.g., to Tensor)

* `testbench.py`:
    * **Function**: Used to test the various functions defined in `data_process.py`.
    * **Purpose**: To ensure that each preprocessing step works as expected and to validate data dimensions, ranges, and types.

## Quick Start

### 1. Installing Dependencies

This project may depend on the following libraries:

* Python 3.x
* NumPy
* PyTorch


It is recommended to read `requirements.txt` (which I will upload later) and install using:

``` bash
pip install -r requirements.txt
``` 

## More
I'm currently working on this project and will be adding more useful features soon. Feel free to contact me if you have any questions!
