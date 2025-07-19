# Advanced Automated Portfolio Allocation System for the Brazilian (B3) Market

This repository contains the source code for a robust, reproducible, and academically grounded research stack for Brazilian equity portfolio allocation, as detailed in the project specification.

The system is designed to be modular, separating signal generation from risk estimation and using a Bayesian framework (Black-Litterman) to synthesize views and produce optimized portfolio weights.

**Status:** Under Development

---

## Setup and Installation

This project is managed using `conda` for environment control and `pip` for installing the local package.

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd b3_alloc_system
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all necessary dependencies. The environment should be named `finance`.
    ```bash
    # Create the environment from the file
    conda env create -f environment.yml

    # Activate the environment
    conda activate finance
    ```
    *Note: The final line in the `environment.yml` file (`- -e .`) also installs the project's own source code in "editable" mode. This means any changes you make to the Python files in `/src/b3alloc` will be immediately available without needing to reinstall.*

3.  **Verify the installation:**
    Once the environment is active, you should be able to run `python` and `import b3alloc` without errors.

---

## Project Structure

-   **/src/b3alloc**: The core Python package containing all logic.
-   **/notebooks**: Jupyter notebooks for research, validation, and reporting.
-   **/config**: YAML configuration files for managing parameters.
-   **/data**: Raw, processed, and intermediate data files.
*   **/scripts**: Standalone Python scripts for running major tasks (e.g., data ingestion, backtesting).
-   **/reports**: Output for generated backtest reports, figures, and artifacts.