#!/bin/bash
#SBATCH --job-name=L2_sal # Job name
#SBATCH --nodes=1 # Run on a single node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai # Run in ai queue
#SBATCH --qos=ai 
#SBATCH --account=ai 
#SBATCH --gres=gpu:tesla_t4:1 
#SBATCH --mem=20G 
#SBATCH --time=1-12:0:0 # Time limit days-hours:minutes:seconds
#SBATCH --output=test-%j.out # Standard output and error log
#SBATCH --mail-type=NONE # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=atekirdag17@ku.edu.tr # Where to send mail

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load anaconda/3.6
module load gcc/9.3.0
source activate elec491

python main.py Eval/test Out/
conda deactivate