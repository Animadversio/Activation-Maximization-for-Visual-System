Note on Image Generator System
========

The programs all starts from `experiment_***.py` in which the experiment program is coded in a main class, which all inherited from the base clasee `ExperimentBase`


One very important factor in the program is the optimizer algorithm. 



**Note**: Search for `__name__ == '__main__'` is a good way to find the start of a program

The target of this rotation is to do multifaceted visualization for neurons in a CNN / a Hierachical visual pathway. 

In the existing algorithm, our method is to get a score for each presented picture. $f(I)$ and use one zeroth order optimization algorithm to optimize the code basing on response. 

`DynamicParameter` class: wrapper of parameter class


## Genetic algorithm

`Genetic` class in `Optimizer`, and the core algorithm part is the `step` function in the `Genetic`

`int(n_conserve)` # important parameter to determine how many parents in last generation is preserved 

**Initialization** `self._init_population = self._random_generator.normal(loc=0, scale=1, size=(self._popsize, 4096))` the initial population has a i.i.d. multivariate standard normal distributed code. 
Which can be improved maybe. 

### The iteration cycle 

**nan dealing** Those samples with invalid score, i.e. will go into next generation to get scored! 
```python
thres_n_nans = min(n_nans, new_size)
            new_samples[-thres_n_nans:] = self._curr_samples[nan_mask][-thres_n_nans:]
            new_genealogy[-thres_n_nans:] = curr_genealogy[nan_mask][-thres_n_nans:]
```

**Q** : Where is the source of `nan` score / output

**Selection** 
The top `self._n_conserve` ones are conserved without mutation. The following `self._n_conserve:thres_n_valid` ones will go through mutation. And Others will be generated


`self._kT = max((np.std(valid_scores) * self._kT_mul, 1e-8))` 
Self adaptive change of temperature `kT`, proportional to $std$ of population score distribution. 

`fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)`
Note, the fitness score is generated using exponential function of the score, with the most favorite sample has the score 1 and all other less than one. 

`self._kT` the temperature parameter dictates the overall selectivity among the scores, i.e. higher `kT` then the scores are more equal. 

Actually, when the scores cluster together, the scores can be quite close, so then maybe the `kT` should be adaptively changed to suit the score. Or we can use rank score instead of raw score to get scale invariant. 


**Mating** 
`new_samples[:thres_n_valid]` : use top `thres_n_valid` best samples as parents and mate! and get the last few samples. 

**Mutation** 
Mutate the middle ones `self._n_conserve:thres_n_valid` 



Record the best sample image in all the history! 
```python
if self._best_score is None or self._best_score < valid_scores[0]:
                self._best_score = valid_scores[0]
                self._best_code = new_samples[0].copy()
```






