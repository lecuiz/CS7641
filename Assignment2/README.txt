Hey there again, instructions for running code and producing plots.

Data:
    stored here: https://gatech.box.com/s/0xbenbungzjmru2146o1dllqp6grk53z
    contains:
        - requirements.txt: python requirements for running script
        - main_ro.py: script for running RO algorithms on three optimization problems
        - main_ro_runners.py: script for parameter optimization of RO algorithms
        - main_nn.py: script for fitting NN to dataset with RO algorithms
        - tic+tac+toe+endgame/: tic tac toe dataset
        - results/: folder containing previously generated figures and data, for cleanliness

Run:
    # Run main to solve optimization problems with RO algos
    'python main_ro.py'
    # Run runners for generating Figures (2, 4, 6) in document
    'python main_ro_runners.py'
    # Run to fit NN to tic, tac, toe dataset
    'python main_nn.py'

Output
    main_ro.py
        4 figures per optimization problem
        e.g.:
            - FlipFlop_1.png
            - FlipFlop_2.png
            - FourPeaks_1.png
            ...
    main_ro_runners.py
        3 folders with run data
            - FourPeaks/
            - OneMax/
            - FlipFlop/
        3 figures for GA
            - FourPeaksGA_mutation_rate_0.2.png
            - FourPeaksGA_mutation_rate_0.5.png
            - FourPeaksGA_mutation_rate_0.7.png
        3 figures for MIMIC
            - OneMaxMIMIC_keep_pct_0.25.png
            - OneMaxMIMIC_keep_pct_0.5.png
            - OneMaxMIMIC_keep_pct_0.65.png
        2 figures for SA
            - FlipFlopSA_fevals.png
            - FlipFlopSA_fitness.png

    main_ro_runners.py
        12 figures stored in datafolder (2 per algorithm)
        4 learning curves plots
        4 training time plots
        4 convergence over iteration plots
        e.g.:
            - tic+tac+toe+endgame/
                    - genetic_alg_learning_curve.png
                    - genetic_alg_training_time.png
                    - genetic_alg_iters.png
                    ...