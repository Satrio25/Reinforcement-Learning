import pygame
import math
import numpy as np
import random
import time

from pygame.constants import BUTTON_X2
ROWS = 10
WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Complete Coverage Path Planning")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (102, 191, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.color == BLACK

    def is_free(self):
        return self.color == WHITE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_barrier(self):
        self.color = BLACK

    def make_path(self):
        self.color = GREEN 
        
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.y, self.x, self.width, self.width))

    def __lt__(self, other):
        return False

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    x, y = pos

    row = y // gap
    col = x // gap

    return row, col

def init_params():
    w1 = (np.random.rand(128, 100))*math.sqrt(2/228)
    b1 = (np.random.rand(128, 1))*math.sqrt(2/100)
    w2 = (np.random.rand(128, 128))*math.sqrt(2/256)
    b2 = (np.random.rand(128, 1))*math.sqrt(2/128)
    w3 = (np.random.rand(4, 128))*math.sqrt(2/132)
    b3 = (np.random.rand(4, 1))*math.sqrt(2/128)
    return w1, b1, w2, b2, w3, b3

def copy_network(w1, b1, w2, b2, w3, b3):
    return w1, b1, w2, b2, w3, b3

def ReLU(Z):
    x, y = Z.shape
    for i in range(x):
        for j in range(y):
            if Z[i][j]>0:
                Z[i][j] = Z[i][j] 
            else:
                Z[i][j] = 0
    return Z

def sigmoid(Z):
    x, y = Z.shape
    for i in range(x):
        for j in range(y):
            Z[i][j] = 1/(1+np.exp(-Z[i][j]))
    return Z
def sigmoid_deriv(Z):
    x, y = Z.shape
    for i in range(x):
        for j in range(y):
            Z[i][j] = (1-Z[i][j])*Z[i][j]
    return Z
def ReLU_deriv(Z):
    x, y = Z.shape
    for i in range(x):
        for j in range(y):
            if Z[i][j]>0:
                Z[i][j] = 1
            else:
                Z[i][j] = 0
    return Z

def forward_prop(x, w1, b1, w2, b2, w3, b3):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = z3
    return z1, a1, z2, a2, z3, a3

def backward_prop(gradient_error, input, z1, a1, z2, a2, z3, a3, w1, w2, w3):
    dw3 = gradient_error.dot((a2.T))
    db3 = gradient_error
    da2 = (w3.T).dot(gradient_error)
    dz2 = da2 * ReLU_deriv(z2)
    dw2 = dz2.dot((a1.T))
    db2 = dz2
    da1 = (w2.T).dot(dz2)
    dz1 = da1 * ReLU_deriv(z1)
    dw1 = dz1.dot((input.T))
    db1 = dz1
    return dw1, db1, dw2, db2, dw3, db3

def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 = w1 + alpha * dw1
    b1 = b1 + alpha * db1    
    w2 = w2 + alpha * dw2  
    b2 = b2 + alpha * db2
    w3 = w3 + alpha * dw3  
    b3 = b3 + alpha * db3    
    return w1, b1, w2, b2, w3, b3


def updateMap(envMap, env, row, col):
    if((row>0 and row<9) and (col>0 and col<9)):
        up = env[row-1][col]
        down = env[row+1][col]
        left = env[row][col-1]
        right = env[row][col+1]
        
        if (envMap[row-1][col])!=1: 
            envMap[row-1][col] = up
        if (envMap[row+1][col])!=1:
            envMap[row+1][col] = down
        if (envMap[row][col+1])!=1:
            envMap[row][col+1] = right
        if (envMap[row][col-1])!=1:
            envMap[row][col-1] = left
        envMap[row][col] = 1
        list_around = np.array([envMap[row-1][col], envMap[row+1][col], envMap[row][col-1], envMap[row][col+1]])
    return envMap, list_around
'''
0 = free cell
1 = visited cell
2 = obstacle 
'''
def cekDone(envMap):
    dimRow, dimCol = envMap.shape
    temp = 0
    for i in range(dimRow):
        for j in range(dimCol):
            if envMap[i][j]==0:
                temp = temp + 1
    if (temp>0):
        return False
    else:
        return True
def info_state(envMap, next_row, next_col):
    if(envMap[next_row][next_col]==2):
        reward = -5
    elif(envMap[next_row][next_col]==1):
        reward = -3
    else:
        reward = 4
    return reward

def next_state(envMap, action, current_row, current_col):
    next_row = current_row
    next_col = current_col
    if (action==0): #UP
        next_row = current_row - 1
    elif (action==1): #DOWN
        next_row = current_row + 1
    elif (action==2): #LEFT
        next_col = current_col - 1
    else: #RIGHT
        next_col = current_col + 1
    #print("(next_row, next_col):"+str(next_row)+","+str(next_col))
    reward = info_state(envMap, next_row, next_col)
    
    if envMap[next_row][next_col]==2:
        next_row = current_row
        next_col = current_col

    return next_row, next_col, reward

def move_state(envMap, w1, b1, w2, b2, w3, b3, choose_action, list_around, explor_rate, current_row, current_col, reward):
    input_fp = (envMap.reshape(100, 1))/2
    z1, a1, z2, a2, z3, a3 = forward_prop(input_fp, w1, b1, w2, b2, w3, b3)
    Qt = a3
    #print("z1")
    #print(z1[0])
    #print("a1")
    #print(a1[0])
    if choose_action >= explor_rate:
        action = np.argmax(Qt)
    else:
        action = np.argmin(list_around)
        #action = random.randrange(0,4)
    
    #action = np.argmin(list_around)
    #print(action)
    next_row, next_col, reward = next_state(envMap, action, current_row, current_col)
    return next_row, next_col, action, Qt, reward

def update_network(episode, reward, envMapNew, envMap, next_row, next_col, current_row, current_col, w1, b1, w2, b2, w3, b3, alpha, gamma, Qt, action):
    input_fp = (envMap.reshape(100,1))/2
    input_fpNew = (envMapNew.reshape(100,1))/2
    z1, a1, z2, a2, z3, a3 = forward_prop(input_fpNew, w1, b1, w2, b2, w3, b3)
    Qt_next = a3

    z1, a1, z2, a2, z3, a3 = forward_prop(input_fp, w1, b1, w2, b2, w3, b3)
    Qt = a3
    
    if(episode==0):
        #cekawal = np.array_equal(input_fp, input_fpNew)
        #print(cekawal)
        print(Qt_next)
        print("------")
        print(Qt)
        cek = np.array_equal(Qt_next, Qt)
        print(cek)
        print("......")
        #print(envMap)
        #print(envMapNew)
        time.sleep(0.2)
    
    #print("-----Qt-----")
    #print(Qt)
    #print("----Qt next----")
    #print(Qt_next)
    #print("reward : "+str(reward))
    gradient_error = np.zeros([4,1])
    gradient_error[action] = (reward + gamma*np.max(Qt_next))-Qt[action]
    #print("----grad error---")
    #print(gradient_error)
    dw1, db1, dw2, db2, dw3, db3 = backward_prop(gradient_error, input_fp, z1, a1, z2, a2, z3, a3, w1, w2, w3)
    w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
    return w1, b1, w2, b2, w3, b3

def copy_matriks(z):
    dimX, dimY = z.shape
    copy = np.zeros([dimX, dimY])
    for i in range(dimX):
        for j in range(dimY):
            copy[i][j] = z[i][j]
    return copy
def DQN(start_row, start_col, env, grid, win, width):
    w1, b1, w2, b2, w3, b3 = init_params()
    
    explor_rate = 1
    alpha = 0.00001
    gamma = 0.9
    maxEps = 100
    for episode in range(maxEps):
        envMap = np.ones([10, 10]) + np.ones([10, 10])
        print("------Episode : "+str(episode)+"-------")
        print("explor rate : "+str(explor_rate))
        current_row = start_row
        current_col = start_col
        done = False
        envMap, list_around = updateMap(envMap, env, current_row, current_col)
        step = 0
        reward = 0
        while not(done) and step<150:
            print(b1[0])
            #print(envMap)
        #while not(done):
            step += 1
            #print("step : "+str(step))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            choose_action = random.random()
            #choose_action = np.argmin(list_around)
            #if(episode==(maxEps-1)):
            if(episode==0):
                spot = grid[current_row][current_col]
                spot.make_path()
                draw(win, grid, ROWS, width)
            #print(envMap)
            #print("sebelum")
            
            next_row, next_col, action, Qt, reward = move_state(envMap, w1, b1, w2, b2, w3, b3, choose_action, list_around, explor_rate, current_row, current_col, reward)
            temp = copy_matriks(envMap)
            #print(temp)
            #print("sebelum tes")
            envMapNew, list_around = updateMap(temp, env, next_row, next_col)
            #print(envMap)
            #print("tes")
            #print(envMapNew)
            #print("testing")
            #print("--------------")
            #print(envMapNew)
            #if(episode==0):
            #    cek = np.array_equal(envMap, envMapNew)
            #    print(cek)
            #    time.sleep(0.2)
            done = cekDone(envMapNew)
            if done:
                reward = 10
            w1, b1, w2, b2, w3, b3 = update_network(episode, reward, envMapNew, envMap, next_row, next_col, current_row, current_col, w1, b1, w2, b2, w3, b3, alpha, gamma, Qt, action)
            envMap = envMapNew
            current_row, current_col = next_row, next_col
            
            #if(episode==(maxEps-1)):
            if(episode==0):
                #print(list_around)
                #print(Qt)
                spot = grid[current_row][current_col]
                spot.make_start()
                draw(win, grid, ROWS, width)
                #print("reward : "+str(reward))
                time.sleep(0.3)
            
        #print(envMap)
        print("step : "+str(step))
        explor_rate = 1-(episode/maxEps)
    return w1, b1, w2, b2, w3, b3, envMap

def main(win, width):
    grid = make_grid(ROWS, width)
    env = np.zeros((ROWS, ROWS))
    start = None
    run = True
    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if pygame.mouse.get_pressed()[0]: #LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start:
                    start = spot
                    start.make_start()
                    start_row = row
                    start_col = col
                elif spot != start:
                    spot.make_barrier()
                    env[row][col]=2
                #print(env)
            elif pygame.mouse.get_pressed()[2]: #RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                env[row][col]=0
                #print(env)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start:
                    w1, b1, w2, b2, w3, b3, envMap = DQN(start_row, start_col, env, grid, win, width)
                    print("learning is done")
                    envMap = np.ones([10, 10]) + np.ones([10, 10])
                    pos_row = start_row
                    pos_col = start_col
                    done = False
                    while not(done):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                        spot = grid[pos_row][pos_col]
                        spot.make_path()
                        envMap, list_around = updateMap(envMap, env, pos_row, pos_col)
                        draw(win, grid, ROWS, width)
                        input_fp = envMap.reshape(100, 1)
                        z1, a1, z2, a2, z3, a3 = forward_prop(input_fp, w1, b1, w2, b2, w3, b3)
                        Qt = a3
                        print(Qt)
                        action = np.argmax(Qt)
                        if action==0:
                            pos_row -= 1
                        elif action==1:
                            pos_row += 1
                        elif action==2:
                            pos_col -= 1
                        else:
                            pos_col += 1
                        spot = grid[pos_row][pos_col]
                        spot.make_start()
                        draw(win, grid, ROWS, width)
                        time.sleep(0.3)
                        done = cekDone(envMap)  
    pygame.quit()

main(WIN, WIDTH)
