# Drone attitude control

Simulating a drone following a trajectory.
Controlling the drone via its attitude and total thrust. Using acados and CasADi 

## Table of Contents

- [Drone attitude control](#drone-attitude-control)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [Usage](#usage)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

   ```sh
   git clone https://github.com/BroilerCompiler/drone-attitude-control.git

2. Navigate to the project directory:

   ```sh
    cd drone-attitude-control

3. Create a virtual environment (optional but recommended):

   ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install the required packages:

   Install acados_template Python package:

   ```sh
    pip install <acados_root>/interfaces/acados_template
    pip install -r requirements.txt

## Usage

   ```sh
    cd src/
    python main.py
