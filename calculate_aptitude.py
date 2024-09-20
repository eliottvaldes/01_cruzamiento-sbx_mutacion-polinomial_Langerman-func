import numpy as np

"""
@breaf: This function is used to calculate the aptitude of the individuals in the population using any Objective Function passed as a parameter
@context: General approach to evaluate the individuals in the population
@param population: np.ndarray -> All population to be evaluated -> F.E: [[2.3, 4.5], [1.4, -0.2]]
@param objective_function: function -> Objective function to evaluate the individuals
@return: np.ndarray -> Aptitude of the individuals in the population (Vector) [9, 5]
"""
def evaluate_population(population: np.ndarray, objective_function) -> np.ndarray:
    # get the number of individuals and variables
    n_individuals, n_var = population.shape
    # create the aptitude vector
    aptitude = np.zeros(n_individuals)
    # loop over the population
    for i in range(n_individuals):
        # get the individual
        individual = population[i]
        # evaluate the individual
        aptitude[i] = objective_function(individual)
    return aptitude



# defining the langermann function
def langermann_function(individual: np.ndarray) -> float:
    # define the constants
    a = np.array([3, 5, 2, 1, 7])
    b = np.array([5, 2, 1, 4, 9])
    c = np.array([1, 2, 5, 2, 3])
    # get the variables of the individual
    x1, x2 = individual
    # calculate the result
    result = 0
    for i in range(len(a)):
        # calculate the first term - cos
        cos_term = np.cos(np.pi * ((x1 - a[i])**2 + (x2 - b[i])**2))
        # calculate the second term - exp
        exp_term = np.exp(((x1 - a[i])**2 + (x2 - b[i])**2) / np.pi)        
        # accumulate the result dividing the cosine by the exponential and multiplying by c[i]
        result += c[i] * (cos_term / exp_term)
    # return the negative result
    return -result

    

if __name__ == '__main__':
    # JUST FOR TESTING
    # GENERAL CONFIGURATIONS
    population = np.array([[2.3 , 4.5], [1.4, -0.2]])
    
    # evaluate the population
    aptitude = evaluate_population(population, langermann_function)
    # get the index of the minimum aptitude
    best_index = np.argmin(aptitude)
    
    print(f'Population: \n{population}\n')
    print(f'Aptitude: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {population[best_index]}')   