# Standard Operating Procedure (SOP) for User Onboarding and Job Execution on CSIS Cluster [cite: 1]

## Purpose: - [cite: 1]
This SOP provides users with a definitive guide to accessing the newly configured CSIS Cluster, understanding its resources, software, and policies, and successfully submitting computational jobs. [cite: 1] This ensures consistent, secure, and efficient use of shared resources. [cite: 1]

## Scope: - [cite: 1]
This procedure applies to all institute/department stakeholders who have been granted access to the aforementioned cluster. [cite: 1]

## Prerequisites: - [cite: 1]
* An approved user account with valid credentials. [cite: 1]
* An SSH client (For Example: - ssh on Linux/macOS, PuTTY/MobaXterm on Windows). [cite: 1]
* Basic familiarity with the Linux CLI mode (Command Line Interface). [cite: 1]

## Definitions: - [cite: 1]
* **Login Node/Master Node:** The entry point for editing files, compiling code, and submitting jobs. [cite: 1]
* **Slurm:** The job scheduler (Simple Linux Utility for Resource Management) manages all compute resources. [cite: 1] Slurm schedules the job and sets priorities for it based on the cluster factors. [cite: 1]
* **Batch Job:** A computational task is submitted to a queue (partition) to run without user interaction. [cite: 1]
* **Compute Node:** Worker nodes where jobs are executed. [cite: 1] Access is allocated by the scheduler. [cite: 1]
* **Environment Modules:** Software that allows users to dynamically load pre-installed applications and their dependencies into their shell environment. [cite: 1]
* **Partition:** A queue of nodes with specific resource limits (max time, cores/GPU). [cite: 1]
* **QoS (Quality of Service):** A set of limits that override or supplement partition limits, often used to grant different priorities or capabilities (e.g., longer runtimes). [cite: 1]
* **SBATCH Script:** A shell script containing directives for Slurm (`#SBATCH` lines) and commands to run your software. [cite: 1]

## Technical Specifications of Compute Node: [cite: 1]
* **OEM & Model:** DELL PowerEdge R7525 [cite: 1]
* **CPU:** 2x AMD EPYC 7742 (128 cores total per node) [cite: 1]
* **RAM:** 256 GB DDR4 memory per node [cite: 1]
* **OS:** Rocky Linux 8.10 [cite: 1]
* **Kernel:** 4.18.0-553.el8_10.x86_64 [cite: 1]
* **NVIDIA Driver:** 570.158.01 [cite: 1]
* **GPU:** 2x NVIDIA A100 80GB GPUs per node [cite: 1]
* **CUDA Cores:** 6,912 per GPU [cite: 1]
* **Tensor Cores:** 432 per GPU [cite: 1]
* **Total VRAM & CUDA Cores per Node:** 160 GB VRAM; 13,824 CUDA cores [cite: 1]
* **Workload Manager:** Slurm 23.11.11 (https://slurm.schedmd.com/) [cite: 1]

## CPU & GPU Specifications Information: [cite: 1]
* Nodes contain AMD EPYC 7502 32-Core Processors (Model 49 Stepping 0), running at 2500 MT/s. [cite: 1]
* Processors support capabilities like AMD-V and No Execute. [cite: 1]
* Cache Information: L1 (2 MB), L2 (16 MB), L3 (128 MB), all Unified and Internal, with Write Back policy. [cite: 1]
* GPUs are NVIDIA A100 80GB PCIe models. [cite: 1]

**To View GPU Specifications in each node:** [cite: 1]
```bash
nvidia-smi
# To see GPU utilization, including memory
nvidia-smi -q | grep -E "(Product Name | FB Memory | Count)"
```
[cite: 1]

**Visual Representation of a Cluster Node** [cite: 1]
* RAM: 256 GB [cite: 1]
* CPU Complex (128 cores) [cite: 1]
* GPU Card #1: NVIDIA A100 80GB [cite: 1]
* GPU Card #2: NVIDIA A100 80GB [cite: 1]

**DISCLAIMER:** The information contained in this document is subject to change without notice. [cite: 1] The department shall not be liable for errors contained herein or for incidental or consequential damages in connection with the performance or use of this manual. [cite: 1]

---

## Procedure: - [cite: 1]

### Step 1: Secure Connection (Login) to the Cluster [cite: 1]
Connect using the following command: [cite: 1]
I. Open a terminal window (or SSH client) on the end-user device. [cite: 1]
II. `ssh <username>@172.24.16.132` [cite: 1]
III. Enter user password and any 2FA token when prompted. [cite: 1]

*Web console:* `https://csis.mn1.cluster_csis1:9090/` or `https://172.24.16.132:9090/` [cite: 1]

### Step 2: Understand the Environment [cite: 1]
Initially, users will be on a master node as they successfully log in. [cite: 1] User is supposed to execute the job by preparing and using Slurm files, not directly on the node's terminal console. [cite: 1] Each user has a 300 GB quota limit for their home directory. [cite: 1] It may be extended as the cluster is getting more disk availability. [cite: 1]

*   **To check the home directory usage details:** `du -sh $HOME` or `du -sh /nfs_home/users/username` [cite: 1]
*   **To check the allotted home directory size details:** `quota -u $USER` or `quota -u /nfs_home/users/username` [cite: 1]

The cluster utilizes a shared, common scratch directory with a total capacity of 1 TB available to all users. [cite: 1] To simplify access, users have a symbolic link placed directly under their home directory that points to their dedicated subdirectory within the common scratch space. [cite: 1] 
*(Use `cd ~/scratch` to get into respective scratch directory)* [cite: 1]

To prevent any single user from monopolizing the limited storage, any data residing in a user's scratch subdirectory that is older than 30 days will be automatically deleted. [cite: 1]

**NB:** If the allotted user quota is near or exceeded with stored data, the user must archive or delete user data to ensure adequate space before new jobs can run or data can be saved. [cite: 1]

### Step 3: File Transfers [cite: 1]
To transfer a File, the user can use SCP to transfer files between the local machine and the cluster. [cite: 1]
a) **To upload files from the local/user-end machine to the cluster:** [cite: 1]
* `scp /<local_directory_path>/<local_file(s)> <username>@172.24.16.132:/$HOME/<remote_directory_path>/` [cite: 1]

b) **To download files from the cluster to the local/user-end machine:** [cite: 1]
* `scp <username>@172.24.16.132:/$HOME/<remote_directory_path>/<remote_file(s)> /<local_directory_path>/` [cite: 1]

*The user may use any of the available/familiar GUI applications (WinSCP, FileZilla) for meeting the file transferring requirements.* [cite: 1]

### Step 4: Explore Available Software Modules and Resources [cite: 1]
A key software stack is pre-installed and managed via Environment Modules. [cite: 1] This simplifies loading complex software and its dependencies. [cite: 1]
Key Available Modules Include: [cite: 1]
* `apps/jupyter/lab`, `apps/miniconda/default`, `apps/nvhpc/24.1`, `apps/cuda/12.8`, `apps/gamess/00` [cite: 1]
* `envs/python/3.9`, `envs/pytorch/1.0`, `envs/tensorflow/1.0` [cite: 1]
* `lang/python/3.13.5 (D)` [cite: 1]
* `mpi/default`, `mpi/mpich-x86_64`, `mpi/openmpi-x86_64`, `mpi/pmix-x86_64` [cite: 1]
* `libraries/fftw/3.3.10-double`, `libraries/hdf5/1.14.3-serial`, `libraries/openblas/0.3.27`, `libraries/boost/1.84.0` [cite: 1]

If the avail list is too long consider trying: [cite: 1]
* `module -default avail` or `ml -d av` to just list the default modules. [cite: 1]
* `module overview` or `ml ov` to display the number of modules for each name. [cite: 1]
* Use `module spider` to find all possible modules and extensions. [cite: 1]
* Use `module keyword key1 key2` to search for all possible modules matching any of the "keys". [cite: 1]

**Short Notes about available Modules:** [cite: 1]
1) **lang/ (Programming Languages):** [cite: 1]
   * `python/3.9`: The Python 3.9 interpreter and core libraries. [cite: 1] Useful for projects that require compatibility with this specific, older version. [cite: 1]
   * `python/3.13.5 (D)`: The default Python 3.13.5 interpreter. [cite: 1] This is the latest version provided by the system. [cite: 1]
2) **envs/ (Pre-configured Environments):** [cite: 1]
   * `pytorch/1.0`: A pre-installed environment for PyTorch 1.0. [cite: 1]
   * `tensorflow/1.0`: A pre-installed environment for TensorFlow 1.0. [cite: 1]
3) **mpi/ (Message Passing Interface Libraries):** [cite: 1]
   * `openmpi-x86_64 (D)`: The default version of the OpenMPI library. [cite: 1]
   * `mpich-x86_64`: The MPICH implementation. [cite: 1]
   * `pmix-x86_64`: The Process Management Interface (PMIx) library. [cite: 1]
4) **libraries/ (Scientific & Numerical Libraries):** [cite: 1]
   * `openblas/0.3.27`: OpenBLAS library, an optimized BLAS implementation. [cite: 1]
   * `LAPACK` and `ScaLAPACK`: Libraries for higher-level linear algebra operations and parallel extensions. [cite: 1]
   * `fftw/3.3.10-double`: The FFTW library for high-performance Fourier transforms. [cite: 1]
   * `hdf5/1.14.3-serial`: The HDF5 library. [cite: 1]
   * `boost/1.84.0`: Boost C++ source libraries. [cite: 1]
5) **apps/ (Applications & Tools):** [cite: 1]
   * `miniconda/default`: Provides the 'conda' package manager. [cite: 1] Essential for creating isolated environments. [cite: 1]
   * `jupyter/lab`: For launching JupyterLab sessions. [cite: 1]
   * **GAMESS**: Quantum chemistry software package for ab initio molecular orbital calculations. [cite: 1]
   * **CUDA 12.8 Toolkit**: NVIDIA's parallel computing platform and programming model. [cite: 1]
   * **NVIDIA HPC SDK 24.1**: Complete suite of high-performance compilers and libraries (nvc++, nvc, nvfortran). [cite: 1]

### Step 5: How to View and Load Available Software Modules [cite: 1]
* `module avail` - Overview of all partitions and node states [cite: 1]
* `module avail python` - To view all available modules [cite: 1]
* `module show lang/python/3.13.5` - To search for specific modules (e.g., python) [cite: 1]
* `module load miniconda/default` - To get information about a specific module [cite: 1]
* `echo $LD_LIBRARY_PATH` - To check the module library paths currently set [cite: 1]
* `module load jupyter/lab` - To load JupyterLab for interactive work [cite: 1]
* `module list` - To see currently loaded modules [cite: 1]
* `module unload jupyter/lab` - To unload a specified module [cite: 1]
* `module purge` - To unload all modules [cite: 1]

*Mention the required module name explicitly for hassle-free setup.* [cite: 1]

### Step 6: How to View Cluster's Workload Manager Specifications & Partitions [cite: 1]
* `sinfo` - Overview of all partitions and node states [cite: 1]
* `sinfo -o "%20P %5D %14F %8z %10m %10c %10G"` - Detailed view: Partition, Nodes, State, Memory, CPUs, GPUs [cite: 1]
* `sinfo -o "%20N %10G"` - To see the physical GPU count per node [cite: 1]

### Step 7: Know the Cluster resources and availability with respect to Partitions/QoS [cite: 1]

**QoS (Quality of Service) and Partition Mapping** [cite: 1]
The user may refer to the table below to know the most effective preset of Slurm for the proposed job. [cite: 1]

| QoS/Partition Name | Priority | Best For | Max Time | Max CPUs | Max (RAM) Memory | Max GPUs | Maximum Concurrent Jobs per QoS/Partition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| debug | 1000 | Quick testing, debugging | 30 min | 16 | 48 GB | 0 GPU | 10 |
| cpu-short | 600 | Standard CPU Jobs | 12 hours | 64 | 96 GB | 0 GPU | 30 |
| cpu-long | 300 | Long-running CPU jobs | 24 hours | 32 | 64 GB | 2 GPUs | 15 |
| gpu-short | 500 | GPU-accelerated jobs | 8 hours | 24 | 96 GB | 1 GPU | 10 |
| gpu-long | 350 | Long-running GPU jobs | 12 hours | 16 | 80 GB | 1 GPU | 6 |
| gpu-1day | 250 | GPU jobs up to 1 Day | 24 hours | 16 | 64 GB | 2 GPUs | 4 |
| gpu-3day | 200 | GPU jobs up to 3 Days | 72 hours | 16 | 64 GB | 1 GPU | 2 |
| bigmem | 400 | Memory-intensive jobs | 8 hours | 64 | 192 GB | 0 GPU | 5 |

[cite: 1]

**Limits Tables:** [cite: 1]

| Maximum Concurrent Jobs per User | Remark |
| :--- | :--- |
| 5 | Each individual user is limited to 5 concurrent jobs. |

[cite: 1]

| Group Name | Maximum Concurrent Jobs per Group | GrpTRES Limits | Remark |
| :--- | :--- | :--- | :--- |
| faculty | 30 | cpu=64; mem=720 GB; gres/gpu=8; | Shared limits across all faculty users to balance concurrency and resources. |

[cite: 1]

**How this works:** [cite: 1]
* MaxJobs=5 means each user can only have 5 running jobs simultaneously. [cite: 1]
* The 6th job will automatically be queued (PD state) with reason "JobHeldUser". [cite: 1]
* As soon as one of the 5 running jobs finishes, the next queued job will start. [cite: 1]
* This applies across all partitions/QoS for that user. [cite: 1]
* Concurrent running jobs limit per faculty group also works in a similar way. [cite: 1]
* To ensure fair resource allocation and prevent system monopolization, the cluster workload manager has been configured with per-user and overall group concurrent job limits, along with Partition/QoS resource utilization restrictions. [cite: 1]
* Partitions/QoS Resources and Executable Job limits may be adjusted appropriately by analyzing the cluster usage on a pro-rata basis. [cite: 1]

**Illustrative Priority Scenarios of Partition/QoS:** [cite: 1]
1. **debug (Weight - 1000):** High (Runs immediately), Medium (Runs within 15 mins), Lower (Runs within 30 mins) [cite: 1]
2. **cpu-short (Weight - 600):** High (Runs within 30 mins), Medium (Runs within 1-2 hours), Lower (May wait 3-4 hours during high demand) [cite: 1]
3. **cpu-long (Weight - 300):** High (Runs within 4-6 hours), Medium (Runs within 8-12 hours), Lower (May wait 24+ hours) [cite: 1]
4. **gpu-short (Weight - 500):** High (Runs immediately to 30 mins), Medium (Runs within 1-2 hours), Lower (May wait 3-4 hours) [cite: 1]
5. **gpu-long (Weight - 350):** High (Runs within 2-4 hours), Medium (Runs within 4-8 hours), Lower (May wait 12+ hours) [cite: 1]
6. **gpu-1day (Weight - 250):** High (Runs within 4-8 hours), Medium (Runs within 12-18 hours), Lower (May wait 24+ hours) [cite: 1]
7. **gpu-3day (Weight - 200):** High (Runs within 8-12 hours), Medium (Runs within 24-48 hours), Lower (May wait 3+ days) [cite: 1]
8. **bigmem (Weight - 400):** High (Runs immediately to 30 mins), Medium (Runs within 2-3 hours), Lower (May wait 6+ hours) [cite: 1]

### Step 8: Job Scheduling and Priority Determination [cite: 1]
**Job Scheduling Mechanism:** The cluster is configured with Slurm's backfill scheduling policy. [cite: 1] This maximizes system utilization by allowing the scheduler to start lower-priority jobs, provided they won't delay higher-priority pending jobs. [cite: 1] Setting reasonably accurate time limits is essential. [cite: 1]

**Job Priority Calculation:** A job's priority score determines its queue position. [cite: 1] The final priority is a weighted sum expressed by the formula: [cite: 1]
`Job_priority = site_factor + (100)*(age_factor) + (0)*(assoc_factor) + (1000)*(fair-share_factor) + (200)*(job_size_factor) + (500)*(partition_factor) + (1000)*(QoS_factor) + SUM(TRES_weight_cpu * TRES_factor_cpu, ...) - nice_factor` [cite: 1]

**Priority Factors:** [cite: 1]
* **Fair-Share Factor (1000):** Most influential factor, based on allocated vs consumed resources. [cite: 1]
* **QoS Factor (1000):** Priority assigned to the requested QoS. [cite: 1]
* **Partition Factor (500):** Priority assigned to the requested partition. [cite: 1]
* **Job Size Factor (200):** Correlates to requested nodes or CPUs. [cite: 1]
* **Age Factor (100):** Boost for older jobs waiting in the queue. [cite: 1]
* **Association Factor (0):** Priority assigned to account/user. [cite: 1]
* **TRES Weights:** Based on requested Trackable RESources. [cite: 1]
* **Nice Factor:** Voluntarily lower job priority. [cite: 1]

### Step 9: Slurm Script Setup Instructions for Users [cite: 1]
1) Create a job script (`my_test_job.sh`) mentioning partition name, resources, and modules. [cite: 1] Use `vi`, `vim`, or `nano`. [cite: 1]
2) **Slurm File Syntax:** [cite: 1]
```bash
#!/bin/bash
#SBATCH --job-name=my_test_job
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00

module load lang/python/3.13.5
module load miniconda3/4.12.0

echo "Starting the test job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
python /path/to/your/script/my_script.py
```
[cite: 1]

3) Make it executable: `chmod +x my_test_job.sh` [cite: 1]
4) Submit the job: `sbatch my_test_job.sh` [cite: 1]

**Best Practices:** [cite: 1]
* Use appropriate QoS levels. [cite: 1]
* Don't over-request hardware resources - it affects the fair share priority...! [cite: 1]
* Request realistic time. [cite: 1]
* Complete jobs promptly. [cite: 1]

### Step 10: To monitor and know about the executed job [cite: 1]
1) **To monitor and manage the job in progress:** [cite: 1]
* `squeue -u $USER` - check status of the job [cite: 1]
* `scontrol show job <jobid>` - know executed job status [cite: 1]
* `sinfo -o "%20P %10a %10c %10m %10G %60"` - show resource utilization [cite: 1]
* `sprio -u $USER` - check job priorities [cite: 1]
* `sshare -u $USER` - check fair share priority status [cite: 1]
* `tail -f test_%j.out` / `tail -f test_%j.err` - check output/error files [cite: 1]
* `scontrol update job <jobid> set <new attribute value>` - change attributes [cite: 1]
* `scontrol hold <jobid>` / `scontrol release <jobid>` - hold/release job [cite: 1]
* `scancel <jobid>` - cancel pending/running job [cite: 1]

2) **To know the completed job, and to know about pending jobs:** [cite: 1]
* `sacct -u $USER` - view job history [cite: 1]
* `cat test_%j.out` / `cat test_%j.err` - check completed output/error files [cite: 1]
* `squeue -t pending -o "%.8i %.12P %.20j %.8u %.2t %.10M %.6D %25R"` - pending jobs reasons [cite: 1]
* `squeue -t pending --start` - estimated start times [cite: 1]
* `seff <jobid>` - detailed efficiency report [cite: 1]

**Quick Reference for One-liner Execution Syntax:** [cite: 1]
* `sbatch --partition=debug --time=00:30:00 --mem=16G script.sh` [cite: 1]
* `sbatch --partition=short --time=04:00:00 --gres=gpu:1 script.sh` [cite: 1]
* `sbatch --partition=long --time=24:00:00 --mem=48G script.sh` [cite: 1]
* `sbatch --partition=bigmem --mem=180G script.sh` [cite: 1]

*(Various Slurm script templates for Basic CPU, GPU, Debug, Long-running CPU, Memory-Intensive, and Multi-GPU jobs using Miniconda are provided as standard examples.)* [cite: 1]

**To Integrate Slurm with Major Development Platforms:** [cite: 1]
1) **VS Code (with Miniconda):** [cite: 1]
   * `source /nfs_home/software/miniconda/etc/profile.d/conda.sh` [cite: 1]
   * `conda activate my_env` [cite: 1]
   * `python my_script.py` [cite: 1]
   * *Use Slurm Dashboard extension to submit/monitor.* [cite: 1]
2) **JupyterLab:** [cite: 1]
   * `module load jupyter/lab` [cite: 1]
   * `jupyter lab --no-browser --port=8855 --ip=0.0.0.0` [cite: 1]
   * Port forwarding: `ssh -L 8855:localhost:8855 username@<cluster_IP>` [cite: 1]
   * *Access in browser with token.* [cite: 1]
3) **Conda Environment for Jupyter:** [cite: 1]
   * `conda install ipykernel` [cite: 1]
   * `python -m ipykernel install --user --name my_env --display-name "My Conda Environment"` [cite: 1]

Users are hereby encouraged to make use of authorized online resources to overcome the learning curve and become familiar with the cluster and its operations. [cite: 1]

### Step 11: Getting Help and Support [cite: 1]
* Users may email their queries or requests for additional module expansions to the support team. [cite: 1]
* In case of any technical error, users are requested to include the corresponding job ID and the relevant section of the slurm-.out file in their email. [cite: 1]
* Support can be reached through the following channels: [cite: 1]
  * Email: vincem@pilani.bits-pilani.ac.in [cite: 1]
  * In-Person Assistance: CSIS Lab (Room 6017). [cite: 1]
