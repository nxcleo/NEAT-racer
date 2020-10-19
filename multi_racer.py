import pygame
import numpy as np
import math
import sympy
import timeit
import neat
import pickle
import random
import os
from multiprocessing import Process, Pool
import time

# Attributes
resolution = np.array((900, 900))
map_size = np.array((2000, 1700))
fps = 60


# Class definitions
class Segment:
    def __init__(self, f, min_x, max_x, direction=1):
        self.x = sympy.symbols('x')
        self.f = sympy.sympify(f)
        self.d1 = sympy.diff(self.f, self.x)
        self.min = min_x
        self.max = max_x
        self.direction = direction


class Car:
    img = pygame.image.load('car.png')

    def __init__(self, racer, loc_x=0.0, loc_y=0.0, direction=0):
        self.loc = np.array([loc_x, loc_y], dtype='float64')
        self.vel = 0
        self.acc = 0
        self.dir = direction
        self.w = 85
        self.h = 42
        self.racer = racer
        self.score = 0
        self.alive = True
        self.intersect = None

        if racer.render:
            self.surface = pygame.Surface((self.w, self.h))
            self.surface.set_colorkey((0, 0, 0))
            self.surface.blit(Car.img , (0, 0))

    @staticmethod
    def get_intersections(map_edge, dir, pos):
        ray_vector = np.array((math.cos(math.radians(dir)), -math.sin(math.radians(dir))))
        pos = np.array(pos)
        p1, p2 = np.array(map_edge[:-1], dtype=np.float32), np.array(map_edge[1:], dtype=np.float32)
        ray_normal = np.array((math.cos(math.radians(dir - 90)), -math.sin(math.radians(dir - 90))), dtype=np.float32)

        intersect = np.dot(p1 - pos, ray_normal) * np.dot(p2 - pos, ray_normal) < 0
        p1, p2 = p1[intersect], p2[intersect]

        if not len(p1):
            return False

        m = np.array(ray_vector[1] / ray_vector[0])
        mp = (np.hsplit(p1, 2)[1] - np.hsplit(p2, 2)[1]) / (np.hsplit(p1, 2)[0] - np.hsplit(p2, 2)[0])
        x = (np.hsplit(p1, 2)[1] - pos[1] + pos[0] * m - np.hsplit(p1, 2)[0] * mp) / (m - mp)
        y = (x - pos[0]) * m + pos[1]

        direction = ((x - pos[0]) > 0) == (ray_vector[0] > 0)
        x, y = x[direction], y[direction]
        if not len(x):
            return False

        dist = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5
        idx = 0
        mini = dist[0]
        for i in range(len(dist)):
            if dist[i] < dist[idx]:
                idx, mini = i, dist[i]
        return (x[idx], y[idx]), mini

    # sensor(screen, map_edge, resolution, car.dir, car.loc, 40)
    def sensor(self, offset=40):
        dat = [self.vel]
        self.intersect = []
        for i in range(-2, 3):
            intersect = Car.get_intersections(self.racer.map_edge, self.dir + i * offset, self.loc)
            self.intersect.append(intersect)
            if intersect:
                dat.append(intersect[1])
                if intersect[1] < 15:
                    self.alive = False
            else:
                dat.append(1000)
        return dat

    def update(self, nn_input):
        self.action_backward(nn_input[0])
        self.action_forward(nn_input[1])
        self.action_left(nn_input[2])
        self.action_right(nn_input[3])

        self.loc += self.vel * np.array([math.cos(math.radians(self.dir)), -math.sin(math.radians(self.dir))])
        self.vel += self.acc - self.vel/15

    def getShape(self):
        return pygame.transform.rotate(self.surface, self.dir)

    def action_forward(self, state):
        self.acc = 1 * state

    def action_backward(self, state):
        self.acc = -1 * state

    def action_right(self, state):
        self.dir -= 2.94398 * state

    def action_left(self, state):
        self.dir += 2.94398 * state




class MultiRacer:
    def __init__(self, resolution, map_size, fps, render=True):

        self.res = resolution
        self.map_size = map_size
        self.fps = fps
        self.max_rounds = 300
        self.rounds = 0
        self.t = timeit.default_timer()
        self.instance_no = random.randint(0, 10000)
        self.render = render
        self.cars = []
        self.generation_count = 1

        self.segments = [Segment("-1.6621*10**-3*x**2+1.2466*x+1.1418*10**3", 65, 700),
                         Segment("8.2850*10**-6*x**3+-1.9061*10**-2*x**2+13.426*x+-1.6999*10**3", 700, 1000),
                         Segment("-8.4507*10**-6*x**3+3.1146*10**-2*x**2+-3.6781*10*x+1.5036*10**4", 1000, 1400),
                         Segment("-4.3464*10**-3*x**2+1.2908*10**1*x+-8.1529*10**3", 1400, 1800),
                         Segment("1.3768*10**-3*x**2+-3.2923*x+2.4654*10**3", 1150, 1800, -1),
                         Segment("1.7167*10**-6*x**3+-4.5457*10**-3*x**2+3.5186*x+-1.4549*10**2", 600, 1150, -1),
                         Segment("2.4654*10**-68*x**3+-1.4557*10**-3*x**2+1.6646*x+2.2531*10**2", 200, 600, -1),
                         Segment("(0.003*x-3.15)**6+100", 200, 1800)
                         ]
        self.map_edge = MultiRacer.generate_map_edge(self.segments)
        self.map_half = np.hsplit(np.array(self.map_edge[:len(self.map_edge) // 2]), 2)
        self.max_score = len(self.map_half[0]) -1
        self.score_prev = 0
        self.p_best = 0

        if self.render:
            pygame.init()
            self.win = pygame.display.set_mode(resolution)
            self.screen = pygame.Surface(map_size + resolution * 2)
            self.screen.fill((255, 255, 255))
            self.map = pygame.Surface(map_size)
            self.map.fill((255, 255, 255))
            pygame.display.set_caption("Racer")
            self.font = pygame.font.SysFont('Consolas', 18)
            self.camera = np.array((0, 0))
            self.score_text = self.font.render('Population Best: 0', False, (0, 0, 0))
            pygame.draw.polygon(self.map, (100, 100, 100), self.map_edge, 0)

    @staticmethod
    def check_camera(camera, keys):
        s = 5
        if keys[pygame.K_w]:
            camera[1] += s
        if keys[pygame.K_s]:
            camera[1] -= s
        if keys[pygame.K_d]:
            camera[0] -= s
        if keys[pygame.K_a]:
            camera[0] += s
        return camera

    @staticmethod
    def generate_map_edge(segments, step=20, gap=85):
        """
        check https://www.desmos.com/calculator/woszzztqne
        :param segments:
        :param step:
        :param gap:
        :return:
        """
        left_edge = []
        right_edge = []
        for seg in segments:
            a = sympy.symbols('a')
            gap_func = seg.d1.subs(seg.x, a) / abs(seg.d1.subs(seg.x, a)) * sympy.sqrt(
                gap ** 2 / (1 + 1 / seg.d1.subs(seg.x, a) ** 2))
            gap_func = sympy.lambdify(a, gap_func, 'numpy')

            normal_func = seg.f.subs(seg.x, a) + -1 / seg.d1.subs(seg.x, a) * (seg.x - a)

            x_values = np.arange(seg.min, seg.max, step)[::seg.direction]
            x_gap = gap_func(x_values)

            left_x = x_values - x_gap
            right_x = x_values + x_gap

            for i in range(len(left_x)):
                if not (np.isnan(left_x[i]) or np.isnan(right_x[i])):
                    if seg.direction == 1:
                        left_edge.append((left_x[i], normal_func.subs(seg.x, left_x[i]).subs(a, x_values[i])))
                        right_edge.append((right_x[i], normal_func.subs(seg.x, right_x[i]).subs(a, x_values[i])))
                    else:
                        right_edge.append((left_x[i], normal_func.subs(seg.x, left_x[i]).subs(a, x_values[i])))
                        left_edge.append((right_x[i], normal_func.subs(seg.x, right_x[i]).subs(a, x_values[i])))
        return left_edge + right_edge[::-1]

    @staticmethod
    def shortest_dist(array_x, array_y, pos, prev):
        lo = max(prev - 3, 0)
        hi = min(prev + 3, len(array_x))

        x = pos[0]
        y = pos[1]
        dist_array = (array_x[lo:hi] - x) ** 2 + (array_y[lo:hi] - y) ** 2
        idx = 0
        dist = dist_array[0][0]
        for i in range(len(dist_array)):
            if dist_array[i][0] < dist:
                dist = dist_array[i][0]
                idx = i
        return dist ** 0.5, idx + lo

    def reset(self):
        self.cars = []
        self.rounds = 0
        self.p_best = 0
        self.generation_count += 1

    def evaluate(self, net):
        car = Car(env, 50, 1200, -45)
        rounds = 0
        while car.alive and rounds <= self.max_rounds and car.score <= self.max_score:
            car.update(net.activate(car.sensor()))
            car.score = MultiRacer.shortest_dist(self.map_half[0], self.map_half[1], car.loc, car.score)[1]
        return car.score**2

    def step(self):
        if not self.cars:
            return
        self.rounds += 1

        if self.render:
            self.camera = MultiRacer.check_camera(self.camera, pygame.key.get_pressed())
            self.screen.blit(self.map, self.res)

        best_car = None
        best_score = -1
        live_count = 0
        for car in self.cars:
            if car.alive:
                live_count += 1

                if self.render:
                    for intersect in car.intersect:
                        if intersect:
                            color = int(200 * (min(intersect[1], 500) / 500))
                            pygame.draw.line(self.screen, (200 - color, color, 0), car.loc + self.res,
                                             intersect[0] + self.res, 3)

                    car_shape = car.getShape()
                    self.screen.blit(car_shape, car.loc - car_shape.get_rect().center + self.res)
                car.score = MultiRacer.shortest_dist(self.map_half[0], self.map_half[1], car.loc, car.score)[1]
                if car.score > best_score:
                    best_car = car
                    best_score = car.score

        if not live_count or self.rounds >= self.max_rounds or best_score >= self.max_score:
            scores = [car.score**2 for car in self.cars]
            self.reset()
            return scores

        if self.render:
            self.win.blit(self.screen, self.camera - best_car.loc - self.res / 2)

            if best_score > self.p_best:
                self.p_best = best_score
            if self.score_prev != self.p_best:
                self.score_text = self.font.render('Population Best: ' + str(self.p_best), False, (0, 0, 0))
                self.score_prev = best_car.score
            self.win.blit(self.score_text, (20, 20))

            pygame.time.delay(max(16 - int((timeit.default_timer() - self.t) * 1000), 0))
            self.t = timeit.default_timer()
            pygame.display.update()

    @staticmethod
    def quit():
        pygame.quit()


def eval_genome(genomes, config):
    env.cars = [Car(env, 50, 1200, -45) for _ in range(len(genomes))]
    nets = [neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            for genome_id, genome in genomes]

    while True:
        for i in range(len(env.cars)):
            if env.cars[i].alive:
                nn_out = nets[i].activate(env.cars[i].sensor())
                env.cars[i].update(nn_out)

        res = env.step()

        if res:
            print(env.generation_count, res)
            save((genomes[res.index(max(res))][1], config), env.instance_no)
            for i in range(len(genomes)):
                genomes[i][1].fitness = res[i]
            break


def save(obj, instance_no):
    with open('genome' + str(instance_no) + ".pkl", 'wb') as f:
        pickle.dump(obj, f)
    print('Genome number', instance_no, "saved.")


def run_saved_genome(instance_no):
    with open('genome' + str(instance_no) + ".pkl", 'rb') as f:
        genome = pickle.load(f)
    env.cars = [Car(env, 50, 1200, -45)]
    net = neat.nn.recurrent.RecurrentNetwork.create(genome[0], genome[1])

    while True:
        if env.cars[0].alive:
            nn_out = net.activate(env.cars[0].sensor())
            env.cars[0].update(nn_out)
        res = env.step()
        if res:
            break


def eva_saved_genome(instance_no):
    with open('genome' + str(instance_no) + ".pkl", 'rb') as f:
        genome = pickle.load(f)
    net = neat.nn.recurrent.RecurrentNetwork.create(genome[0], genome[1])
    print(env.evaluate(net))


env = MultiRacer(resolution, map_size, fps, True)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, 'config-feedforward')

p = neat.Population(config)
#
# winner = p.run(eval_genome)
# print()
run_saved_genome(4263)


