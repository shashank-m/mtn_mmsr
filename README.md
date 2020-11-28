To use the repo:
1. python3 -m venv mtn
2. source mtn/bin/activate
3. git clone https://github.com/shashank-m/mtn_mmsr.git
4. cd mtn_mmsr
5. pip install .

To run the code :
1. python3 optimise.py 'L-BFGS-B'
2. python3 sgd-optim.py



obj_func folder has objective.py which contains the mmsr problem formulation.

optimise.py is where the mmsr objective function is used along with scipy to find solutions.
sgd-optim.py has the sgd implementation.