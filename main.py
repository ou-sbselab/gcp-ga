from multiprocessing.pool import ThreadPool
import multiprocessing as mpc
from time import time as timer
from urllib2 import urlopen

import array
import numpy
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import argparse

parser = argparse.ArgumentParser(description='Executes Cloud-GA')
parser.add_argument('--remote',
                    help='Call GCP instead of local.',
                    action='store_true')

url = "https://us-central1-parallelea.cloudfunctions.net/ea-test2"

def evalOneMax(individual):
  return sum(individual),

def evalWeights(individual):
  equation_inputs = [4,-2,3.5,5,-11,-4.7]
  s = 0.0
  for i in range(len(individual)):
    s += individual[i] * equation_inputs[i]
  return s,

def evalRemoteWeights(individual):
  indv = ""
  for i in range(len(individual)):
    indv += indv + "&x%d=%f" % (i, individual[i])
  indv_url = url + "?" + indv
  
  try:
    response = urlopen(indv_url)
    fit = float(response.read())
    return fit,
  except Exception as e:
    return -9999.99,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.random)
#toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalWeights)#evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main(remote):
  pop = toolbox.population(n=1000)#300)
  CXPB, MUTPB = 0.5, 0.2
  pool = mpc.Pool(processes=12)

  if remote:
    fitnesses = pool.map(evalRemoteWeights, pop)
    #fitnesses = list(ThreadPool(20).imap_unordered(evalRemoteWeights, pop))
  else:
    fitnesses = list(map(toolbox.evaluate, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

  fits = [ind.fitness.values[0] for ind in pop]

  gen = 0
  while gen < 50:#1000:
  #while max(fits) < 100 and gen < 1000:
    gen += 1
    print("Generation %d" % gen)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Crossover
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < CXPB:
        toolbox.mate(c1, c2)
        del c1.fitness.values
        del c2.fitness.values

    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    if remote:
      #fitnesses = list(ThreadPool(20).imap_unordered(evalRemoteWeights, invalid_ind))
      fitnesses = pool.map(evalRemoteWeights, pop)
    else:
      fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2/length - mean**2)**0.5

    print("* Min: %s" % min(fits))
    print("* Max: %s" % max(fits))
    print("* Avg: %s" % mean)
    print("* Std: %s" % std)
    best = tools.selBest(pop, 1)[0]
    print("* Best: %s, %s, %s" % (best, best.fitness.values, evalWeights(best)))

  print "Done."
  best_ind = tools.selBest(pop, 1)[0]
  print("Best individual: %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args.remote)


""" WORKING 
gens = 50
pop  = 100

url = "https://us-central1-parallelea.cloudfunctions.net/ea-test2"
gen = 0
indvs = ["%s?gen=%d&indv=%d" % (url, gen, i) for i in range(pop)]

def fetch_url(url):
  try:
    response = urlopen(url)
    return url, response.read(), None
  except Exception as e:
    return url, None, e

start = timer()
results = ThreadPool(20).imap_unordered(fetch_url, indvs)
for url, html, error in results:
  if error is None:
    #print("%r fetched in %ss" % (url, timer() - start))
    print("[%s fetched in %ss]" % (url, timer() - start))
    print("Response: %s" % html)
  else:
    print("Error fetching %r: %s" % (url, error))
print("Elapsed time: %s" % (timer() - start,))
"""

