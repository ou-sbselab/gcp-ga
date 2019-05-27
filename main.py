from multiprocessing.pool import ThreadPool
import multiprocessing as mpc
from time import time as timer
import math

import array
import numpy
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import argparse

import asyncio
import sys
from typing import IO
import urllib.error
import urllib.parse
import aiofiles
import aiohttp
from aiohttp import ClientSession

import os
import sys
#from subprocess import call
from subprocess import PIPE, run

os.environ['SDL_AUDIODRIVER']            = 'dsp' # fix ALSA error
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'   # Hide PyGame messages
os.environ['SDL_VIDEODRIVER']            = 'dummy' # Run pygame headless


parser = argparse.ArgumentParser(description='Executes Cloud-GA')
parser.add_argument('--remote',
                    help='Call GCP instead of local.',
                    action='store_true')
parser.add_argument('--gens',
                    help='Number of generations to run.',
                    type=int,
                    default=2)
parser.add_argument('--pop_size',
                    help='Population size per generation.',
                    type=int,
                    default=5)
parser.add_argument('--save_frames',
                    help='Write frames to file.',
                    action='store_true')
parser.add_argument('--seed',
                    help='Seed value sent across files.',
                    type=int,
                    default=1)

args = parser.parse_args()
from bouncing_balls import BouncyBalls

url = 'https://us-central1-parallelea.cloudfunctions.net/ea-test-pymunk'

# Optimize Ackley function
def evalAckley(individual): # indv: [x,y] where -5 < x,y < 5 
  x = individual[0]
  y = individual[1]
  val = -20.                                            * \
        math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - \
        math.exp(0.5 * (math.cos(2. * math.pi * x)      + \
                        math.cos(2. * math.pi * y)))    + \
        math.e + 20.
  return val,

# Draw two random lines with the intent of maximizing the number of balls 
# on the screen
def eval2DPhysics(individual, last=False): 
  os.environ['SDL_AUDIODRIVER']            = 'dsp' # fix ALSA error
  os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'   # Hide PyGame messages
  if not last:
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run pygame headless
  else:
    os.environ['SDL_VIDEODRIVER'] = 'x11'   # Run pygame headfull?

#  shell_command = 'python3 bouncing_balls.py --x0 %f --y0 %f --x1 %f --y1 %f --x2 %f --y2 %f --x3 %f --y3 %f' % (individual[0], individual[1], individual[2], individual[3], individual[4], individual[5], individual[6], individual[7])
#  call(shell_command, shell=True, stdin=None, stdout=None, stderr=None)
#  l1 = (individual[0],individual[1],individual[2],individual[3])
#  l2 = (individual[4],individual[5],individual[6],individual[7])
#  game = BouncyBalls(l1,l2)
#  return game.run(),
  shell_cmd = ['python3','bouncing_balls.py',\
               '--seed', str(args.seed),    \
               '--x0', str(individual[0]), \
               '--y0', str(individual[1]), \
               '--x1', str(individual[2]), \
               '--y1', str(individual[3]), \
               '--x2', str(individual[4]), \
               '--y2', str(individual[5]), \
               '--x3', str(individual[6]), \
               '--y3', str(individual[7])]
  result = run(shell_cmd, stdout=PIPE, stderr=PIPE, text=True)
  # Need to use STDERR as the return because PyMunk doesn't seem to allow suppression of its
  # loading message.
  return int(result.stdout.strip()), 

def evalOneMax(individual):
  return sum(individual),

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

#toolbox.register("attr_float", random.uniform, -5.0, 5.0)
toolbox.register("attr_int", random.randint, 0, 600) #make pair if not square resolution
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 8)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

 
pool = mpc.Pool(8)
toolbox.register("map", pool.map)
toolbox.register("evaluate", eval2DPhysics)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=600, indpb=0.2)
#toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)



# AsyncIO functions for interacting with Google Cloud Functions
async def fetchCF(indv: str, session: ClientSession, **kwargs) -> str:
  resp = await session.request(method="GET", url=indv, **kwargs)
  resp.raise_for_status()
  html = await resp.text()
  return html

# Dispatch the CFs and return fitness
async def callCF(indv: str, session: ClientSession, **kwargs) -> float:
  try:
    html = await fetchCF(indv=indv, session=session, **kwargs)
  except (
    aiohttp.ClientError,
    aiohttp.http_exceptions.HttpProcessingError,
  ) as e:
    #return "Error processing [%s]" % indv
    print(indv,e)
    return -9999.99,
  else:
    return int(html),

# Called via GA
async def evalAsync(pop: set, **kwargs) -> None:
  async with ClientSession() as session:
    tasks = []
    for p in pop:
      #print(p)
      indv_url = '%s?seed=%d&x0=%d&y0=%d&x1=%d&y1=%d&x2=%d&y2=%d&x3=%d&y3=%d' % \
                 (url,args.seed,int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5]),int(p[6]),int(p[7]))
      tasks.append(
        callCF(indv=indv_url,session=session,**kwargs)
      )
    return await asyncio.gather(*tasks)

# Write out invalid indices per generation
def writepop(gen,pop):
    with open('checkvals.txt','a') as f:
        f.write('Generation %d\n' % gen)
        for p in pop:
            f.write(str(p)+'\n')
        f.write('======================\n')


def main(remote, gens, pop_size, save_frames=False):
  pop = toolbox.population(n=pop_size)
  CXPB, MUTPB = 0.5, 0.2

  if remote:
    fitnesses = asyncio.run(evalAsync(pop))
    print("Fitnesses:",fitnesses)
  else:
    fitnesses = toolbox.map(toolbox.evaluate, pop)

  # Copy fitnesses to population
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
  fits = [ind.fitness.values[0] for ind in pop]

  gen = 0
  while gen < gens:
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

    # Mutation
    for mutant in offspring:
      if random.random() < MUTPB:
        toolbox.mutate(mutant)
        for m in mutant:
            if m < 0:    m = 0
            if m > 600:  m = 600
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    if remote: # Turn population into CF URLs and call remote functions
      fitnesses = asyncio.run(evalAsync(invalid_ind))
      print(fitnesses)

    # Otherwise, evaluate locally via MPC
    else:
      fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    print("%d invalid indices evaluated" % len(invalid_ind))

    # Print statistics
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2/length - mean**2)**0.5

    print("* Min: %s" % min(fits))
    print("* Max: %s" % max(fits))
    print("* Avg: %s" % mean)
    print("* Std: %s" % std)
    best = tools.selBest(pop, 1)[0]
    print("* Best: %s, %s" % (best, best.fitness.values))#, eval2DPhysics(best, True)))

    #writepop(gen,invalid_ind)

  print("Done.")
  best_ind = tools.selBest(pop, 1)[0]
  print("Best individual: %s, %s, %s" % (best_ind, best_ind.fitness.values, eval2DPhysics(best_ind, True)))

if __name__ == '__main__':
  assert sys.version_info >= (3, 7), "Requires Python 3.7+"
  main(args.remote, args.gens, args.pop_size, args.save_frames)
