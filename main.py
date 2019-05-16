from multiprocessing.pool import ThreadPool
import multiprocessing as mpc
from time import time as timer
#from urllib2 import urlopen
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

# 107m31.432s

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

args = parser.parse_args()
#if not args.remote:
from bouncing_balls import BouncyBalls

url = 'https://us-central1-parallelea.cloudfunctions.net/ea-test-pymunk'
#url = "https://us-central1-parallelea.cloudfunctions.net/ea-test-ackley"
#url = "https://us-central1-parallelea.cloudfunctions.net/ea-test2"

def evalAckley(individual): # indv: [x,y] where -5 < x,y < 5 
  x = individual[0]
  y = individual[1]
  val = -20.                                            * \
        math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - \
        math.exp(0.5 * (math.cos(2. * math.pi * x)      + \
                        math.cos(2. * math.pi * y)))    + \
        math.e + 20.
  return val,

def eval2DPhysics(individual): #TBD - do something with indiv!
  os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run pygame headless
  game = BouncyBalls()
  game.run()
  return random.random(),

def evalOneMax(individual):
  return sum(individual),

def evalWeights(individual):
  equation_inputs = [4,-2,3.5,5,-11,-4.7]
  s = 0.0
  for i in range(len(individual)):
    s += individual[i] * equation_inputs[i]
  return s,

async def evalRemoteWeights(individual):
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

toolbox.register("attr_float", random.uniform, -5.0, 5.0)
#toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

 
pool = mpc.Pool()
toolbox.register("map", pool.map)

toolbox.register("evaluate", eval2DPhysics)#evalAckley)#evalWeights)#evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

async def fetchCF(indv: str, session: ClientSession, **kwargs) -> str:
  resp = await session.request(method="GET", url=indv, **kwargs)
  resp.raise_for_status()
  html = await resp.text()
  return html

async def callCF(indv: str, session: ClientSession, **kwargs) -> float:
  try:
    html = await fetchCF(indv=indv, session=session, **kwargs)
  except (
    aiohttp.ClientError,
    aiohttp.http_exceptions.HttpProcessingError,
  ) as e:
    #return "Error processing [%s]" % indv
    return -9999.99,
  else:
    if html == 'Done.':
      return 1.0,#float(html),
    else:
      return random.uniform(0.0,0.9),

async def evalAsync(pop: set, **kwargs) -> None:
  async with ClientSession() as session:
    tasks = []
    for p in pop:
      tasks.append(
        callCF(indv=p,session=session,**kwargs)
      )
    return await asyncio.gather(*tasks)

def main(remote, gens, pop_size, save_frames=False):
  pop = toolbox.population(n=pop_size)
  CXPB, MUTPB = 0.5, 0.2
  #url = 'https://us-central1-parallelea.cloudfunctions.net/ea-test2' --> global now
  #pool = mpc.Pool(processes=12)

  if remote:
    # Turn population into CF URLs
    pop_urls = []
    for indv in pop:
      pop_urls.append('%s?x0=%f&x1=%f' % (url, indv[0], indv[1]))
      #pop_urls.append('%s?x0=%f&x1=%f&x2=%f&x3=%f&x4=%f&x5=%f' % (url,indv[0],indv[1],indv[2],indv[3],indv[4],indv[5]))
    fitnesses = asyncio.run(evalAsync(pop=pop_urls))

    #fitnesses = pool.map(evalRemoteWeights, pop)
    #fitnesses = list(ThreadPool(20).imap_unordered(evalRemoteWeights, pop))
  else:
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    #fitnesses = list(map(toolbox.evaluate, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

  fits = [ind.fitness.values[0] for ind in pop]

  gen = 0
  while gen < gens:
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
        for m in mutant:
            if m < -5.0: m = -5.0
            if m > 5.0:  m = 5.0
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    if remote:
      #fitnesses = list(ThreadPool(20).imap_unordered(evalRemoteWeights, invalid_ind))
      #fitnesses = pool.map(evalRemoteWeights, pop)
      # Turn population into CF URLs
      pop_urls = []
      for indv in invalid_ind:#pop:
        pop_urls.append('%s?x0=%f&x1=%f' % (url, indv[0], indv[1]))
       # pop_urls.append('%s?x0=%f&x1=%f&x2=%f&x3=%f&x4=%f&x5=%f' % (url,indv[0],indv[1],indv[2],indv[3],indv[4],indv[5]))
      fitnesses = asyncio.run(evalAsync(pop=pop_urls))
    else:
      fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
      #fitnesses = map(toolbox.evaluate, invalid_ind)
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
    print("* Best: %s, %s, %s" % (best, best.fitness.values, eval2DPhysics(best)))

  print("Done.")
  best_ind = tools.selBest(pop, 1)[0]
  print("Best individual: %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == '__main__':
  assert sys.version_info >= (3, 7), "Requires Python 3.7+"
  main(args.remote, args.gens, args.pop_size, args.save_frames)

  #8m2.337s


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

# real    3m51.628s
