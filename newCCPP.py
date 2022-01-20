from textwrap import indent
import pygame
import math
import numpy as np
import random
import time
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

def whatState(env, row, col, action):
    if (env[row-1][col])==0:
        if(env[row][col-1])==0:
            if(env[row+1][col])==0:
                if(env[row][col+1])==0:
                    if action==0:
                        state=0
                    elif action==1:
                        state=1
                    elif action==2:
                        state=2
                    else:
                        state=3
                if(env[row][col+1])==2:
                    if action==0:
                        state=4
                    elif action==1:
                        state=5
                    elif action==2:
                        state=6
                    else:
                        state=7

            if(env[row+1][col])==2:
                if(env[row][col+1])==0:
                    if action==0:
                        state=8
                    elif action==1:
                        state=9
                    elif action==2:
                        state=10
                    else:
                        state=11
                if(env[row][col+1])==2:
                    if action==0:
                        state=12
                    elif action==1:
                        state=13
                    elif action==2:
                        state=14
                    else:
                        state=15
        if(env[row][col-1])==2:
            if(env[row+1][col])==0:
                if(env[row][col+1])==0:
                    if action==0:
                        state=16
                    elif action==1:
                        state=17
                    elif action==2:
                        state=18
                    else:
                        state=19
                if(env[row][col+1])==2:
                    if action==0:
                        state=20
                    elif action==1:
                        state=21
                    elif action==2:
                        state=22
                    else:
                        state=23

            if(env[row+1][col])==2:
                if(env[row][col+1])==0:
                    if action==0:
                        state=24
                    elif action==1:
                        state=25
                    elif action==2:
                        state=26
                    else:
                        state=27
                if(env[row][col+1])==2:
                    if action==0:
                        state=28
                    elif action==1:
                        state=29
                    elif action==2:
                        state=30
                    else:
                        state=31
    
    if (env[row-1][col])==2:
        if(env[row][col-1])==0:
            if(env[row+1][col])==0:
                if(env[row][col+1])==0:
                    if action==0:
                        state=32
                    elif action==1:
                        state=33
                    elif action==2:
                        state=34
                    else:
                        state=35
                if(env[row][col+1])==2:
                    if action==0:
                        state=36
                    elif action==1:
                        state=37
                    elif action==2:
                        state=38
                    else:
                        state=39

            if(env[row+1][col])==2:
                if(env[row][col+1])==0:
                    if action==0:
                        state=40
                    elif action==1:
                        state=41
                    elif action==2:
                        state=42
                    else:
                        state=43
                if(env[row][col+1])==2:
                    if action==0:
                        state=44
                    elif action==1:
                        state=45
                    elif action==2:
                        state=46
                    else:
                        state=47
        if(env[row][col-1])==2:
            if(env[row+1][col])==0:
                if(env[row][col+1])==0:
                    if action==0:
                        state=48
                    elif action==1:
                        state=49
                    elif action==2:
                        state=50
                    else:
                        state=51
                if(env[row][col+1])==2:
                    if action==0:
                        state=52
                    elif action==1:
                        state=53
                    elif action==2:
                        state=54
                    else:
                        state=55

            if(env[row+1][col])==2:
                if(env[row][col+1])==0:
                    if action==0:
                        state=56
                    elif action==1:
                        state=57
                    elif action==2:
                        state=58
                    else:
                        state=59
                if(env[row][col+1])==2:
                    if action==0:
                        state=60
                    elif action==1:
                        state=61
                    elif action==2:
                        state=62
                    else:
                        state=63
    return state
        
def info_state(env, row, col):
    if((0<row<=ROWS) and (0<col<=ROWS)):
        if (env[row][col] == 2):
            reward = -10
            in_state = False
        elif (env[row][col] == 1):
            reward = -5
            in_state = True
        else:
            reward = 5
            in_state = True
    else:
        reward = -10
        in_state = False
    return reward, in_state
def move(envMap, action, env, row, col):
    new_row = row
    new_col = col
    if(action == 0): #UP
        new_row = row - 1
    elif (action == 1): #DOWN
        new_row = row + 1
    elif (action == 2): #LEFT
        new_col = col - 1
    else: #RIGHT
        new_col = col + 1
    reward, in_state = info_state(envMap, new_row, new_col)
    if (envMap[new_row][new_col]==2):
        new_row = row
        new_col = col
    return reward, new_row, new_col, in_state

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
def q_learning(q_table, env, start_row, start_col, grid, win, width,pg):
    explor_rate=1
    learn_rate=0.001
    discount_rate=0.95

    num_episode=1000
    
    for episode in range(num_episode):
        envMap = np.ones([10, 10]) + np.ones([10, 10])
        action = 0 #DOWN
        row=start_row
        col=start_col
        done = False
        envMap, list_around = updateMap(envMap, env, row, col)
        step=0
        print("-------Episode "+str(episode)+"-------")
        reward = 1
        while step<=(ROWS*ROWS) and not(done):
            #print(step)
            stateEnv=whatState(env, row, col, action)
            step+=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            if(episode==(num_episode-1)):
            #if(episode==0):
                spot = grid[row][col]
                spot.make_path()
                draw(win, grid, ROWS, width)
            choose_action = random.random()
            if(reward<0):
                action = np.argmax(q_table[stateEnv][:])
            else:
                if(pg%2==0):
                #action = random.randrange(0, 4)
                    action = np.argmin(list_around)
                else:
                    min = list_around[0]
                    ind = 0
                    for i in range(4):
                        temp = list_around[i]
                        if temp<=min:
                            min = temp
                            ind = i
                    action = ind


            
            reward, new_row, new_col, in_state = move(envMap, action, env, row, col)
            new_stateEnv =whatState(env, new_row, new_col, action)
            envMap, list_around = updateMap(envMap, env, new_row, new_col)
            done = cekDone(envMap)
            if done:
                reward = 20
            q_table[stateEnv][action]=(q_table[stateEnv][action]*(1-learn_rate))+(learn_rate*(reward+discount_rate*np.max(q_table[new_stateEnv][:])))

            row, col = new_row, new_col
            if(episode==(num_episode-1)):
            #if(episode==0):
                spot = grid[row][col]
                spot.make_start()
                draw(win, grid, ROWS, width)
                #print("reward : "+str(reward))
                time.sleep(0.3)
        explor_rate = 0.01+(1-0.01)*np.exp(-0.001*episode)
    return q_table, envMap

def cekCoverage(envMap):
    dimX, dimY = envMap.shape
    free_cell_count = 0
    visited_cell_count = 0
    for i in range(dimX):
        for j in range(dimY):
            if envMap[i][j] == 0:
                free_cell_count += 1
            elif envMap[i][j] == 1:
                visited_cell_count += 1
            else:
                pass
    coverage = visited_cell_count/(visited_cell_count+free_cell_count)
    return coverage*100



def main(win, width):
    grid = make_grid(ROWS, width)
    env = np.zeros((ROWS, ROWS))
    total_state = 64
    total_action = 4
    q_table = np.random.rand(total_state, total_action)
    start = None
    run = True
    
    state_visited=np.zeros((ROWS,ROWS))
    
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
                    #start_state = (start_row*ROWS)+(start_col+1)
                    
                elif spot != start:
                    spot.make_barrier()
                    env[row][col]=2
                    #state_visited[row][col] = 1
                #print(env)
            elif pygame.mouse.get_pressed()[2]: #RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                env[row][col]=0
                #state_visited[row][col] = 0
                #print(env)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start:
                    coverage_list = []
                    q_tableList = []
                    for i in range(1):
                        q_table, envMap = q_learning(q_table, env, start_row, start_col, grid, win, width, i)
                        #time.sleep(10)
                        #print(envMap)
                        temp = cekCoverage(envMap)
                        print(temp)
                        print("learning "+str(i+1)+" done")
                        coverage_list.append(temp)
                        q_tableList.append(q_table)
                        print(coverage_list)
                        
                    #maxChoose = np.argmax(coverage_list[:])
                    #q_table = q_tableList[maxChoose]
                    max = coverage_list[0]
                    ind = 0
                    for i in range(len(coverage_list)):
                        temp = coverage_list[i]
                        if temp>=max:
                            max = temp
                            ind = i
                    q_table = q_tableList[ind]
                    action = ind
                    print("learning is done")
                    #for i in range(total_state):
                    #    print("state " + str(i))
                    #    print(q_table[i])
                    pos_row = start_row
                    pos_col = start_col
                    stateEnv = whatState(env, pos_row, pos_col, 0)
                    while True:
                        before_pr = pos_row
                        before_pc = pos_col
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                        spot = grid[pos_row][pos_col]
                        spot.make_path()
                        draw(win, grid, ROWS, width)

                        action = np.argmax(q_table[stateEnv][:])
                        if action==0:
                            pos_row -= 1
                        elif action==1:
                            pos_row += 1
                        elif action==2:
                            pos_col -= 1
                        else:
                            pos_col += 1
                        if (envMap[pos_row][pos_col]==2):
                            pos_row = before_pr
                            pos_col = before_pc
                        spot = grid[pos_row][pos_col]
                        spot.make_start()
                        draw(win, grid, ROWS, width)
                        time.sleep(0.3)
                        stateEnv = whatState(env, pos_row, pos_col, action)
    pygame.quit()

main(WIN, WIDTH)
