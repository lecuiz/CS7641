Hey there, instructions for running code and producing plots.

Data:
    stored here: https://gatech.box.com/s/08jh2zlzau92r1yqqf4qqzr3s7gvie94
    contains:
        - requirements.txt: python requirements for running script
        - main.py: main script for running and generating plots
        - tic+tac+toe+endgame/: tic tac toe dataset
        - breast+cancer+wisconsin+diagnostic/: breast cancer dataset

Run:
    # for tic tac toe dataset
    'python main.py tic'
    # for breast cancer dataset
    'python main.py diag'

Output:
    4 figures per ML algorithm
    Figures and tables are stored in respective data folders
    e.g.:
        - tic+tac+toe+endgame/
            - ModelResults.csv
            - ModelResults_hyperparam.png
