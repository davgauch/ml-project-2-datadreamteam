Here’s the completed markdown for your instructions with the missing section filled in:

---

# Transfer the data (and code) to SCITAS

1. **Clone the GitHub repository:**
    - Clone the repository from GitHub using the following command:
    
    ```bash
    git clone <repository_url>
    ```

2. **Download the data and add it to the repository folder under `/data`:**
    - Download the necessary data from the specified source.
    - Once downloaded, move or copy the data into the `/data` directory inside your local cloned repository folder.

3. **Connect to EPFL's VPN:**
    - To access SCITAS (EPFL's HPC cluster), you must connect to EPFL's VPN. Follow the instructions available here:
    [EPFL VPN setup](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/vpn-clients-available/)
    - If you are using a VPN client, make sure to install and configure it before connecting.

4. **Connect to the SCITAS cluster using SSH:**
    - Open a terminal and connect to the cluster using SSH with your EPFL credentials (replace `<GASPAR_ID>` with your EPFL ID):
    
    ```bash
    ssh <GASPAR_ID>@izar.hpc.epfl.ch
    ```

5. **Transfer the local folder to your scratch folder on the cluster (30-day deletion policy):**
    - Once connected to the cluster, you can transfer your local repository folder (including data) to your scratch space on SCITAS using `rsync`. Make sure to replace `<PATH_OF_REPO_FOLDER>` with the local path to your repository folder and `<GASPAR_ID>` with your EPFL ID:
    
    ```bash
    rsync -azP <PATH_OF_REPO_FOLDER> <GASPAR_ID>@izar.hpc.epfl.ch:/scratch/izar/<GASPAR_ID>
    ```

    - **Note:** This will copy the entire repository folder (including data) to the `/scratch/izar/<GASPAR_ID>` directory on the cluster. 

6. **If you only want to transfer the code without data (just to update the code):**
    - If you don't want to transfer the large dataset and only need to sync the code, you can exclude the `/data` folder during the transfer. Use the following `rsync` command to skip the data folder:
    
    ```bash
    rsync -azP --exclude 'data' <PATH_OF_REPO_FOLDER> <GASPAR_ID>@izar.hpc.epfl.ch:/scratch/izar/<GASPARID>
    ```
    
    - This command will transfer only the code and exclude the contents of the `data` folder.
Here is a reformulated version of your instructions:

---

# Setting Up the Environment

1. **Load the necessary modules:**

   First, load the required modules for GCC, Python, and CUDA using the following command:

   ```bash
   module load gcc python cuda
   ```

2. **Create and activate the Python virtual environment:**

   Create a new virtual environment in your scratch space (replace `<GASPAR_ID>` with your actual EPFL ID), and then activate it:

   ```bash
   virtualenv --system-site-packages /scratch/izar/<GASPAR_ID>/venvs/pytorch-env
   source /scratch/izar/<GASPAR_ID>/venvs/pytorch-env/bin/activate
   ```

3. **Reload modules if necessary:**

   If you encounter issues with missing dependencies or modules, reload the modules again:

   ```bash
   module load gcc python cuda
   ```

4. **Install required libraries:**

   Install the necessary Python libraries listed in your `requirements.txt` file:

   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

5. **Deactivate the environment:**

   Once you're done working, deactivate the virtual environment:

   ```bash
   deactivate
   ```

---

# Running the Job

First update the GASPAR_ID in run_scitas.slurm, the second line should be:

#SBATCH --chdir=/scratch/izar/<GASPAR_ID>


Then, to submit your job to the SCITAS cluster, use the following command:

```bash
sbatch run_scitas.slurm
```