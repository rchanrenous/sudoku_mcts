import numpy as np
import random
from math import sqrt
import math
import copy
import collections
import time
import multiprocessing as mp

## Sudoku generator functions
# from: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python

# pattern for a baseline valid solution
def pattern(r,c, size, base):
  return (base*(r%base)+r//base+c)%size

# randomize rows, columns and numbers (of valid base pattern)
def shuffle(s):
  return random.sample(s,len(s))
  
## Hash table
Dx = Dy = 16
hashTable = []
for k in range (17):
    l = []
    for i in range (Dx):
        l1 = []
        for j in range (Dy):
            l1.append (random.randint (0, 2 ** 64))
        l.append (l1)
    hashTable.append (l)
    
class Sudoku:
  # 3x3 or 4x4= so i 1 to 16. 
  """
  this class generate sodoku 
  """
  def __init__(self, size=9, rm_rate=0.5):
    self.size = size
    self.base = int(sqrt(size)) # 3, 4, 5, 6
    self.grid = np.zeros((size, size)) # board games
    self.h = 0

    self.move_to_play = collections.deque()

    rBase = range(self.base) 
    rows  = [ g*self.base + r for g in shuffle(rBase) for r in shuffle(rBase) ] 
    cols  = [ g*self.base + c for g in shuffle(rBase) for c in shuffle(rBase) ]
    nums  = shuffle(range(1,self.base*self.base+1))

    self.grid = np.array([ [nums[pattern(r,c,size,self.base)] for c in cols] for r in rows ])

    nb_remove = int(rm_rate * size**2) 
    remove = random.sample(range(1,size*size), nb_remove)
    for i in remove:
      self.grid[i//size][i%size] = 0

    self.domain = [[list(range(1, self.size + 1)) if self.grid[k][j] == 0 else [self.grid[k][j]] for j in range(size)] for k in range(size)] # 2D array

    for i in range(size):
      for j in range(size):
        if self.grid[i][j]!=0:
          self.fc(i, j, self.grid[i][j])
          self.h ^= hashTable[0][i][j]
          self.h ^= hashTable[self.grid[i][j]][i][j] 
    
    self.empty_cells = np.where(self.grid.flatten() == 0)[0].tolist()

  # debugg function
  def check_domain(self):
    for i in range(self.size):
      for j in range(self.size):
        if self.grid[i][j] != 0:
          neigh = self.neighbor(i,j)
          for ii,jj in neigh:
            if self.grid[i][j] in self.domain[ii][jj]:
              # print(self.grid)
              self.display_grid()
              print("Incorrect domain for:", i,j, "and", ii,jj)
              return False
    return True

  # return number of possible moves
  def nb_moves(self):
    list_sum = [sum(el) for el in [[len(l) for l in ll] for ll in self.domain]]
    return (sum(list_sum))
  
  # returns the score of the game
  def score(self):
    return (self.size**2 - len(self.empty_cells)) 

  # debugg function that checks if the grid is consistent
  def consistent(self, log=False):
    for i in range(self.size):
      for j in range(self.size):
        if self.grid[i,j] != 0:
          neigh = self.neighbor(i,j)
          for t in neigh:
            if self.grid[i,j] == self.grid[t]:
              if log:
                print("ijt",i,j,t)
              print("FALSE")
              return False
    return True

  # domain revision
  def fc(self,i,j,val):
    for k in range(self.size):
      if k != i and k != j:
        if val in self.domain[i][k]:
          self.domain[i][k].remove(val)
        if val in self.domain[k][j]:
          self.domain[k][j].remove(val)
      elif k == i and k != j:
        if val in self.domain[i][k]:
          self.domain[i][k].remove(val)
      elif k != i and k == j:
        if val in self.domain[k][j]:
          self.domain[k][j].remove(val)

    # remove val from the domains of the xbox
    starti = int((i)//self.base) * self.base
    endi = starti + self.base
    startj = int(j//self.base) * self.base
    endj = startj + self.base
    for ii in range(starti, endi):
      for jj in range(startj, endj):
        if (ii != i or jj != j) and val in self.domain[ii][jj]:
          self.domain[ii][jj].remove(val)

  # update function for maximal inference playing function
  def update(self,i,j,val):
    neigh = self.neighbor(i,j)
    for ii,jj in neigh:
      if val in self.domain[ii][jj]:
        self.domain[ii][jj].remove(val)
        if len(self.domain[ii][jj]) == 1:
          # print("LOC:",(ii,jj), "val:", val, "domain:", self.domain[ii][jj][0])
          self.move_to_play.append((ii,jj,self.domain[ii][jj][0]))
      if (ii,jj,val) in self.move_to_play:
        self.move_to_play.remove((ii,jj,val))

  # play val at (i,j) and revise the domains
  def play(self,i,j,val):
    # put value in coordinate i, j
    self.grid[i,j] = val
    self.domain[i][j] = [val]
    # remove val from the domains of the variables that have a common constraint
    self.fc(i,j,val)

    # print("play: {}, {}, {}".format(i, j, val))
    self.empty_cells.remove(self.coordinate_to_index(i, j))

  # play val at (i,j), revises the domains, and plays the moves with unique possible values
  def play_max_inference(self,i,j,val):
    index= self.coordinate_to_index(i, j)
    if index in self.empty_cells:
      self.grid[i,j] = val
      self.domain[i][j] = [val]
      # remove val from the domains of the variables that have a common constraint
      # and play the cells with single value domain
      # print("play: {}, {}, {}".format(i, j, val))
      self.empty_cells.remove(index)
      self.update(i,j,val)

  # play val at (i,j), revise the domains and updates the hash
  def play_hash (self, i, j, val):
    self.h = self.h ^ hashTable [0] [i] [j]
    self.h = self.h ^ hashTable [val] [i] [j]
    
    self.grid[i,j] = val
    self.domain[i][j] = [val]
    # remove val from the domains of the variables that have a common constraint
    self.fc(i,j,val)

    # print("play: {}, {}, {}".format(i, j, val))
    self.empty_cells.remove(self.coordinate_to_index(i, j))

  # returns list of neighbors of cell (i,j)
  def neighbor(self,i,j):
    L = []
    # square of the cell
    starti = int((i)//self.base) * self.base
    endi = starti + self.base
    startj = int(j//self.base) * self.base
    endj = startj + self.base

    for ii in range(starti, endi):
      for jj in range(startj, endj):
        if ii!=i or jj!=j:
          L.append((ii,jj))
    
    for ii in range(self.size):
      if ii < starti or ii >= endi:
        L.append((ii,j))

    for jj in range(self.size):
      if jj < startj or jj >= endj:
        L.append((i,jj))
    return L

  def play_fc(self, i, j, val):
    pass

  def play_ac(self, i, j, val):
    pass
  
  # checks if the game is over
  def terminal(self):
    if self.empty_cells == []:
      return True
    for r in self.domain:
      for l in r:
        if l == []:
          return True
    return False

  # converts index to coordinates
  def index_to_coordinate(self, i):
    return (i//self.size, i%self.size)

  # converts coordinates to index
  def coordinate_to_index(self, i, j):
    return i*self.size + j

  # random playing function
  def playout(self, random_fun):
    while(True):
      if self.terminal():
        return self.score()
      else:
        i, j, val = random_fun(self)
        self.play(i, j, val)

  # random maximal inference playing function
  def playout_max_inference(self, random_fun):
    while(True):
      if self.terminal():
        return self.score()
      else:
        i, j, val = random_fun(self)
        self.move_to_play.append((i,j,val))
        while self.move_to_play:
          i ,j, val = self.move_to_play.popleft()
          # print("deque status before play:", self.move_to_play)
          self.play_max_inference(i,j,val)
  
  # vusualization of the neighbor of (i,j)
  def display_neighbor(self, i, j):
    L = self.neighbor(i,j)
    # L = [t for t in gen]
    for i in range(self.size):
      if i%self.base == 0:
        print("-- "*(self.size + self.base//2 + 1))
      for j in range(self.size):
        if j%self.base == 0:
          print("|", end=" ")
        print(self.grid[i][j] if (i,j) not in L else '{}{}{}'.format('\033[31m',self.grid[i][j],'\033[0m'), end=" ")
        if self.grid[i][j] < 10:
          print(end=" ")
      print("|")
    print("-- "*(self.size + int(self.base/2) + 1))

  # display the grid
  def display_grid(self):
    for i in range(self.size):
      if i%self.base == 0:
        print("-- "*(self.size + self.base//2 + 1))
      for j in range(self.size):
        if j%self.base == 0:
          print("|", end=" ")
        print(self.grid[i][j], end=" ")
        if self.grid[i][j] < 10:
          print(end=" ")
      print("|")
    print("-- "*(self.size + int(self.base/2) + 1))
    
    
##################
# playing function

def play(S,i,j,val):
    # put value in coordinate i, j
    S.grid[i,j] = val
    S.domain[i][j] = [val]
    # remove val from the domains of the variables that have a common constraint
    S.fc(i,j,val)
    # print("play: {}, {}, {}".format(i, j, val))
    S.empty_cells.remove(S.coordinate_to_index(i, j))


def play_max_inference(S,i,j,val):
    index = S.coordinate_to_index(i, j)
    if index in S.empty_cells:
      S.grid[i,j] = val
      S.domain[i][j] = [val]
      # remove val from the domains of the variables that have a common constraint
      # and play the cells with single value domain
      S.empty_cells.remove(index)
      S.update(i,j,val)

def play_hash (S, i, j, val):
    # col = int (S.grid [i] [j])
    # if col != 0:
    #     S.h = S.h ^ hashTable [col] [i] [j]
    S.h = S.h ^ hashTable [0] [i] [j]
    S.h = S.h ^ hashTable [val] [i] [j]
    # S.h = S.h ^ hashTurn

    S.grid[i,j] = val
    S.domain[i][j] = [val]
    # remove val from the domains of the variables that have a common constraint
    S.fc(i,j,val)

    # print("play: {}, {}, {}".format(i, j, val))
    S.empty_cells.remove(S.coordinate_to_index(i, j))

def play_max_inference_hash(S,i,j,val):
    S.h = S.h ^ hashTable [0] [i] [j]
    S.h = S.h ^ hashTable [val] [i] [j]

    index = S.coordinate_to_index(i, j)
    if index in S.empty_cells:
      S.grid[i,j] = val
      S.domain[i][j] = [val]
      # remove val from the domains of the variables that have a common constraint
      # and play the cells with single value domain
      S.empty_cells.remove(index)
      S.update(i,j,val)

######################
## playout functions

# random playing function
def playout(S, play_fun, random_fun):
    while(True):
      if S.terminal():
        return S.score()
      else:
        i, j, val = random_fun(S)
        #S.play(i, j, val)
        play_fun(S,i,j,val)

# random maximal inference playing function
def playout_max_inference(S, play_fun, random_fun):
    while(True):
      if S.terminal():
        return S.score()
      else:
        i, j, val = random_fun(S)
        S.move_to_play.append((i,j,val))
        while S.move_to_play:
          i ,j, val = S.move_to_play.popleft()
          # print("deque status before play:", self.move_to_play)
          #S.play_max_inference(i,j,val)
          play_fun(S,i,j,val)
        
  
######################
## random move functions

# picks randomly the cell to play then picks randomly the value from its domain
def random_cells(board):
  can_play_indices = board.empty_cells
  i, j = board.index_to_coordinate(random.choice(can_play_indices))
  val = random.choice(board.domain[i][j])
  return i,j,val

# picks randomly a move from the set of possible values for each cell 
def random_values(board):
  empty_cells_coordinate = [board.index_to_coordinate(index) for index in board.empty_cells]
  list_possible = [[coord[0], coord[1], a] for coord in empty_cells_coordinate for a in board.domain[coord[0]][coord[1]]]
  i, j, val = random.choice(list_possible)
  return i, j, val


#####################
## experiments functions

# general mcts solving function
def mc_solve_attempt(q, S, mc_algo, playout_fun, play_fun, random_fun, n_playouts, cst=0.4):
  while not S.terminal():
    i, j, val, S2 = mc_algo(S, n_playouts, playout_fun, play_fun, random_fun, cst)
    if i != -1:
      play_fun(S, i, j, val)
    else:
      S = S2
      q.put(1)
      return
  if S.score() == S.size**2:
    q.put(1)
    return
  q.put(0)
  return

# mcts algo testing function
def mc_test(time_budget, number_pbs, size, rm_rate, mc_algo, playout_fun, play_fun, random_fun, n_playouts, cst=0.4, seed=1234):
  n_solved = 0
  q = mp.Queue()
  random.seed(seed)
  t_start = time.time()
  for i in range(number_pbs):
    S = Sudoku(size, rm_rate)
    # a = 1
    start = time.time()
    while time.time() - start < time_budget:
        S_copy = copy.deepcopy(S)
        p = mp.Process(target=mc_solve_attempt, args=(q, S_copy, mc_algo, playout_fun, play_fun, random_fun, n_playouts))
        p.start()
        p.join(abs(time_budget-(time.time()-start)))
        if p.is_alive():
            p.terminate()
        sc = q.get() if not q.empty() else 0
        if sc == 1:
          n_solved += 1
          break
    print(f"Problem: {i}, solved: {sc}, time:{time.time()-start}")
  print("Solved:", n_solved, "time:", time.time()-t_start, "\n\n")
  return n_solved, time.time()-t_start

######################
## FLAT mc

def flat_max_inference(board, n, playout_fun, play_fun, random_fun, cst=0.4): #2, 3, 4
    # n_playouts = 0
    moves = board.empty_cells
    max_score = board.size**2
    # print(moves)
    bestScore = 0
    bestMove = 0
    bestVal = 0
    bestI = 0
    bestJ = 0
    for m in range (len(moves)):
        index_cell = moves[m]
        i, j = board.index_to_coordinate(index_cell)
        for val in board.domain[i][j]:
          sum = 0
          for nn in range (n):
              b = copy.deepcopy (board)
              b.play_max_inference(i, j, val)
              # play_fun(b, i, j, val)
              while b.move_to_play:
                ii ,jj, vall = b.move_to_play.popleft()
                b.play_max_inference(ii,jj,vall)
                # play_fun(b, i, j, val)
              #r = b.playout_max_inference(random_fun)
              r = playout_fun(b, play_fun, random_fun)
              if r == max_score: # win
                return -1, -1, 0, b # number of playouts to win
              sum = sum + r
          if sum > bestScore:
              bestScore = sum
              bestMove = m
              bestVal = val
              bestI = i
              bestJ = j
    return bestI, bestJ, bestVal, None #moves[bestMove]

def flat(board, n, playout_fun, play_fun, random_fun, cst=0.4): #2, 3, 4
    moves = board.empty_cells
    max_score = board.size**2
    bestScore = 0
    bestMove = 0
    bestVal = 0
    bestI = 0
    bestJ = 0
    for m in range (len(moves)):
        index_cell = moves[m]
        i, j = board.index_to_coordinate(index_cell)
        for val in board.domain[i][j]:
          sum = 0
          for nn in range (n):
              b = copy.deepcopy (board)
              b.play (i, j, val)

              #r = b.playout(random_fun)
              r = playout_fun(b, play_fun, random_fun)
              if r == max_score: # win
                return -1, -1, 0, b # number of playouts to win
              sum = sum + r
          if sum > bestScore:
              bestScore = sum
              bestMove = m
              bestVal = val
              bestI = i
              bestJ = j
    return bestI, bestJ, bestVal, None #moves[bestMove]
  
########################
## UCT

MaxLegalMoves = 16*16*16
Table = {}

# exploration constant
cst_uct = 0.4

def add (board):
    # nplayouts = [0.0 for x in range (MaxLegalMoves)]
    # nwins = [0.0 for x in range (MaxLegalMoves)]
    # Table [board.h] = [0, nplayouts, nwins]
    nplayouts = [0.0 for x in range (len(board.empty_cells)*16)]
    nwins = [0.0 for x in range (len(board.empty_cells)*16)]
    Table [board.h] = [0, nplayouts, nwins]

def look (board):
    return Table.get (board.h, None)

def UCT (board, playout_fun, play_fun, random_fun, cst=0.4):
    board_size = board.size
    max_score = board_size**2
    if board.terminal ():
        return board.score (), copy.deepcopy(board)
    t = look (board)
    # print(board.h)
    if t != None:
        bestValue = -1000000.0
        best_move = (0, 0, 0)
        best_hash_index = 0
        # for m_move in board.empty_cells:
        for mm in range(len(board.empty_cells)):
            coord1, coord2 = board.index_to_coordinate(board.empty_cells[mm])
            for val_play in board.domain[coord1][coord2]:
              val = 100000.0
              index = mm*16+val_play-1
              if t [1] [index] > 0:
                  Q = t [2] [index] / t [1] [index]
                  val = Q + cst * sqrt (math.log (t [0]) / t [1] [index])
              if val > bestValue:
                  bestValue = val
                  best_move = (coord1, coord2, val_play)
                  best_hash_index = index

        play_fun (board, best_move[0], best_move[1], best_move[2])
        res, b_res = UCT (board, playout_fun, play_fun, random_fun)
        if res == max_score:
          return res, copy.deepcopy(b_res)
        t [0] += 1
        t [1] [best_hash_index] += 1
        t [2] [best_hash_index] += res/max_score
        return res, None
    else:
        add (board)
        #return board.playout (random_fun), copy.deepcopy(board)
        return playout_fun (board, play_fun, random_fun), copy.deepcopy(board)

def BestMoveUCT (board, n, playout_fun, play_fun, random_fun, cst=0.4):
    global Table
    Table = {}
    board_size = board.size
    max_score = board_size**2
    for i in range (n):
        b1 = copy.deepcopy (board)
        res, b_res = UCT (b1, playout_fun, play_fun, random_fun, cst)
        if res == max_score:
          return -1,-1,-1, copy.deepcopy(b_res)
    t = look (board)
    moves = board.empty_cells
    first_index = moves[0]
    first_coord1, first_coord2 = board.index_to_coordinate(first_index)
    first_val = board.domain[first_coord1][first_coord2][0]

    first_hash = first_val-1 

    best = (first_coord1, first_coord2, first_val)
    bestValue = t[1][first_hash]
    # for index in moves:
    for mm in range(len(moves)):
        coord1, coord2 = board.index_to_coordinate(moves[mm])
        for val in board.domain[coord1][coord2]:
          index = mm*16+val-1
          if (t [1] [index] > bestValue):
              bestValue = t [1] [index]
              best = (coord1, coord2, val)
    return best[0], best[1], best[2], None
