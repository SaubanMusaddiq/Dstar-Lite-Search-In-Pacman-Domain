# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    parent_map = {}
    init_state = problem.getStartState()
    frontier = util.Stack()
    frontier_set = set()
    explored_set = set()
    frontier.push(init_state)
    while(True):

        if(frontier.isEmpty()):
            print "Failure"
            return []

        state = frontier.pop()
        if(problem.isGoalState(state)):
            break

        explored_set.add(state)
        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            if(nextState not in explored_set):
                parent_map[nextState] = (state, action)
                frontier.push(nextState)
    bt_state = state
    path = []
    while (bt_state != init_state):
        bt_state, action = parent_map[bt_state]
        path.append(action)

    path.reverse()
    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    init_state = problem.getStartState()
    if(problem.isGoalState(init_state)):
        return []

    parent_map = {}
    frontier = util.Queue()
    frontier_set = set()
    explored_set = set()
    frontier.push(init_state)
    frontier_set.add(init_state)

    while(True):
        if(frontier.isEmpty()):
            print "Failure"
            return []
        state = frontier.pop()
        frontier_set.remove(state)
        explored_set.add(state)
        if(problem.isGoalState(state)):
            break

        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            if(nextState not in explored_set and nextState not in frontier_set):
                parent_map[nextState] = (state, action)
                frontier.push(nextState)
                frontier_set.add(nextState)
    bt_state = state
    path = []
    while (bt_state != init_state):
        bt_state, action = parent_map[bt_state]
        path.append(action)
    path.reverse()
    return path

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    init_state = problem.getStartState()

    parent_map = {}
    cost_map = {}
    frontier = util.PriorityQueue()
    frontier_set = set()
    explored_set = set()
    frontier.push(init_state,0)
    frontier_set.add(init_state)

    path_cost = 0
    cost_map[init_state] = path_cost
    while(True):
        if(frontier.isEmpty()):
            print "Failure"
            return []

        state = frontier.pop()
        path_cost = cost_map[state]
        if(problem.isGoalState(state)):
            break
        frontier_set.remove(state)
        explored_set.add(state)
        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            cost = cost + path_cost
            if(nextState not in explored_set and nextState not in frontier_set):
                cost_map[nextState] = cost
                parent_map[nextState] = (state, action)
                frontier.push(nextState, cost)
                frontier_set.add(nextState)
            elif(nextState in frontier_set and cost < cost_map[nextState]):
                cost_map[nextState] = cost
                parent_map[nextState] = (state, action)
                frontier.update(nextState, cost)

    bt_state = state
    path = []
    while (bt_state != init_state):
        bt_state, action = parent_map[bt_state]
        path.append(action)

    path.reverse()
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    init_state = problem.getStartState()
    parent_map = {}
    cost_map = {}
    frontier = util.PriorityQueue()
    frontier_set = set()
    explored_set = set()
    frontier.push(init_state,0)
    frontier_set.add(init_state)

    path_cost = 0
    cost_map[init_state] = path_cost
    while(True):
        if(frontier.isEmpty()):
            print "Failure"
            return []

        state = frontier.pop()
        path_cost = cost_map[state]
        if(problem.isGoalState(state)):
            break
        frontier_set.remove(state)
        explored_set.add(state)
        for successor in problem.getSuccessors(state):
            nextState, action, cost = successor
            g_cost = path_cost + cost
            h_cost = heuristic(nextState, problem)
            f_cost = g_cost + h_cost
            if(nextState not in explored_set and nextState not in frontier_set):
                cost_map[nextState] = g_cost
                parent_map[nextState] = (state, action)
                frontier.push(nextState, f_cost)
                frontier_set.add(nextState)
            elif(nextState in frontier_set and g_cost < cost_map[nextState]):
                cost_map[nextState] = g_cost
                parent_map[nextState] = (state, action)
                frontier.update(nextState, h_cost)

    bt_state = state
    path = []
    while (bt_state != init_state):
        bt_state, action = parent_map[bt_state]
        path.append(action)

    path.reverse()
    print(path)
    return path


def dStarLiteSearch(problem, heuristic=nullHeuristic):
    dstar = DStarLite(problem)
    path = dstar.move_to_goal()
    # path = [p for p in dstar.move_to_goal()]
    # print(path)
    return path
# self.frontier.push(self.goal, self.calculate_key(self.goal))
# self.back_trace[self.goal] = None


from collections import deque
from functools import partial



class DStarLite(object):
    def __init__(self, problem, view_range=2):
        # Init the graphs
        # self.graph = AgentViewGrid(graph.width, graph.height)
        # self.real_graph: SquareGrid = graph
        self.view_range = view_range
        self.prob = problem
        self.back_pointers = {}
        self.G_VALS = {}
        self.RHS_VALS = {}
        self.Km = 0
        self.position = problem.getStartState()
        self.goal = problem.getGoalState()
        self.frontier = util.PriorityQueue()
        self.frontier.push(self.goal, self.calculate_key(self.goal))
        self.back_pointers[self.goal] = None
        print("init")

    def calculate_rhs(self, node):
        lowest_cost_neighbour = self.lowest_cost_neighbour(node)
        self.back_pointers[node] = lowest_cost_neighbour[0]
        return self.lookahead_cost(node, lowest_cost_neighbour)

    def lookahead_cost(self, node, neighbour):
        # return self.g(neighbour) + self.graph.cost(neighbour, node)
        neighbour,_,_ = neighbour
        return self.g(neighbour) + 1

    def lowest_cost_neighbour(self, node):
        cost = partial(self.lookahead_cost, node)
        # print(node)
        # return min(self.graph.neighbors(node), key=cost)
       # print(self.prob.getSuccessors(node))
       # print(self.position)
        return min(self.prob.getSuccessors(node), key=cost)

        # neighbors = [neighbour for neighbour,_,_ in self.prob.getSuccessors(node)]
        # return min(neighbors, key=cost)
        # # for succ in self.prob.getSuccessors(node):
        #     neighbour,action,cost = succ



    def g(self, node):
        return self.G_VALS.get(node, float('inf'))

    def rhs(self, node):
        return self.RHS_VALS.get(node, float('inf')) if node != self.goal else 0

    def heuristic(self, a, b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def calculate_key(self, node):
        g_rhs = min([self.g(node), self.rhs(node)])

        return (
            g_rhs + self.heuristic(node, self.position) + self.Km,
            g_rhs
        )

    def update_node(self, node):
        if node != self.goal:
            self.RHS_VALS[node] = self.calculate_rhs(node)
        # self.frontier.delete(node)
        if self.g(node) != self.rhs(node):
            # self.frontier.put(node, self.calculate_key(node))
            self.frontier.update(node, self.calculate_key(node))

    def update_nodes(self, nodes):
        [self.update_node(n) for n in nodes]

    def compute_shortest_path(self):
        #print(self.position)
        last_nodes = deque(maxlen=10)
        while not self.frontier.isEmpty() and (self.frontier.nsmallest(1)[0][0] < self.calculate_key(self.position) or self.rhs(self.position) != self.g(self.position)):
            k_old = self.frontier.nsmallest(1)[0][0]
            node = self.frontier.pop()
            last_nodes.append(node)
            if len(last_nodes) == 10 and len(set(last_nodes)) < 3:
                raise Exception("Fail! Stuck in a loop")
            k_new = self.calculate_key(node)
            if k_old < k_new:
                self.frontier.push(node, k_new)
            elif self.g(node) > self.rhs(node):
                self.G_VALS[node] = self.rhs(node)
                neighbors = [neighbour for neighbour,_,_ in self.prob.getSuccessors(node)]
                self.update_nodes(neighbors)
            else:
                self.G_VALS[node] = float('inf')
                neighbours = [neighbour for neighbour,_,_ in self.prob.getSuccessors(node)]
                self.update_nodes(neighbours + [node])
            # print("5",self.frontier.heap)
            # print(self.frontier.nsmallest(1)[0][0] , self.calculate_key(self.position))


        return self.back_pointers.copy(), self.G_VALS.copy()

    def move_to_goal(self):
        # print("start", self.position)
        path = []
        actions = []
        # observation = self.real_graph.observe(self.position, self.view_range)
        #
        # walls = self.graph.new_walls(observation)
        # self.graph.update_walls(walls)

        # print(self.prob.walls,1)
        # print(self.position)
        new_walls = self.prob.update_walls(self.position, self.view_range)

        # print(self.prob.walls)

        self.compute_shortest_path()
        last_position = self.position
        # return

        # yield self.position
        path.append(self.position)

        while self.position != self.goal:
            if self.g(self.position) == float('inf'):
                raise Exception("No path")

            self.position,action,cost = self.lowest_cost_neighbour(self.position)
            new_walls = self.prob.update_walls(self.position, self.view_range)
            actions.append(action)

            # observation = self.real_graph.observe(self.position, self.view_range)
            # new_walls = self.graph.new_walls(observation)
            # print(self.position)
            # print(self.prob.walls)

            if new_walls:
                # self.graph.update_walls(new_walls)
                self.Km += self.heuristic(last_position, self.position)
                last_position = self.position
                self.update_nodes({node for wallnode in new_walls
                                   for node, action, cost in self.prob.getSuccessors(wallnode)})
                # for node in self.prob.getSuccessors(wallnode):
                #     nextState, action, cost = successor
                #
                self.compute_shortest_path()
            # yield self.position
            # return
        return actions

def calcstar(snode, goal, walls):
    sQueue = util.PriorityQueue()
    cnode = pnode(snode.getstate(), [], 0)
    aset = {cnode}
    fringelist = [(cnode.getstate()[0]+1,cnode.getstate()[1]), (cnode.getstate()[0]-1,cnode.getstate()[1]),
    (cnode.getstate()[0],cnode.getstate()[1]-1), (cnode.getstate()[0],cnode.getstate()[1]+1)]
    dir = ["East", "West", "South", "North"]
    for i in range(0, len(fringelist)):
        aset.add(fringelist[i][0])
        if fringelist[i] not in walls:
            wlist = cnode.getacts()[:]
            wlist.append(dir[i])
            newnode=pnode(fringelist[i], wlist, cnode.getcost()+1)
            sQueue.push(newnode, newnode.getcost())
    while cnode.getstate() != goal:
        if sQueue.isEmpty():
            return none
        cnode = sQueue.pop() 
        fringelist = [(cnode.getstate()[0]+1,cnode.getstate()[1]), (cnode.getstate()[0]-1,cnode.getstate()[1]),
        (cnode.getstate()[0],cnode.getstate()[1]-1), (cnode.getstate()[0],cnode.getstate()[1]+1)]
        dir = ["East", "West", "South", "North"]
        for i in range(0, len(fringelist)):
            if fringelist[i] not in aset:
                aset.add(fringelist[i])
                if fringelist[i] not in walls:
                    wlist = cnode.getacts()[:]
                    wlist.append(dir[i])
                    newnode=pnode(fringelist[i], wlist, cnode.getcost()+1)
                    hval = util.manhattanDistance(newnode.getstate(), goal)
                    sQueue.push(newnode, newnode.getcost()+hval)
    return cnode.getacts()
        


class pnode:
    def __init__(self, nstate, acts, cost):
        self.state = nstate
        self.acts = acts
        self.cost = cost


    def getstate(self):
        return self.state

    def getacts(self):
        return self.acts

    def getcost(self):
        return self.cost

def astar2(problem, view_range=1):
    goal= problem.goal
    cstate = problem.startState
    wlist = []
    knownwalls = []
    cost = 0
    walls = problem.walls
    for i in range(0, walls.height):
        knownwalls.append((0, i))
        knownwalls.append((walls.width, i))
    for i in range(0, walls.width):
        knownwalls.append((i, 0))
        knownwalls.append((i, walls.height))
    cnode = pnode(cstate, wlist, 0)
    #hval=heuristic(cnode.getstate(), problem)
    sQueue = util.PriorityQueue() 
    directions = calcstar(cnode, goal, knownwalls)
    while not problem.isGoalState(cnode.getstate()):
        pos = cnode.getstate()
        for i in range(max(1,pos[0]-view_range),min(walls.width,pos[0]+view_range+1)):
            for j in range(max(1,pos[1]-view_range),min(walls.height,pos[1]+view_range+1)):
                if walls[i][j]==True:
                    knownwalls.append((i,j))
        successors = problem.getSuccessors(cnode.getstate())
#        qlist = []
#        for successor in successors: #adding perceivable walls
#            qlist.append(successor[0])
#        if ((cnode.getstate()[0],cnode.getstate()[1]+1) not in qlist) and ((cnode.getstate()[0],cnode.getstate()[1]+1) not in knownwalls):
#            knownwalls.append((cnode.getstate()[0],cnode.getstate()[1]+1))
#        if ((cnode.getstate()[0],cnode.getstate()[1]-1) not in qlist) and ((cnode.getstate()[0],cnode.getstate()[1]-1) not in knownwalls):
#            knownwalls.append((cnode.getstate()[0],cnode.getstate()[1]-1))
#        if ((cnode.getstate()[0]+1,cnode.getstate()[1]) not in qlist) and ((cnode.getstate()[0]+1,cnode.getstate()[1]) not in knownwalls):
#            knownwalls.append((cnode.getstate()[0]+1,cnode.getstate()[1]))
#        if ((cnode.getstate()[0]-1,cnode.getstate()[1]) not in qlist) and ((cnode.getstate()[0]-1,cnode.getstate()[1]) not in knownwalls):
#            knownwalls.append((cnode.getstate()[0]-1,cnode.getstate()[1]))
        direction = directions.pop(0)
        t = 0
        if direction=="North":
            newloc = (cnode.getstate()[0], cnode.getstate()[1]+1)
        if direction=="West":
            newloc = (cnode.getstate()[0]-1, cnode.getstate()[1])
        if direction=="South":
            newloc = (cnode.getstate()[0], cnode.getstate()[1]-1)
        if direction=="East":
            newloc = (cnode.getstate()[0]+1, cnode.getstate()[1])
        if newloc in knownwalls:
            directions = calcstar(cnode, goal, knownwalls)
        else:
            wlist.append(direction)
            cost = cost+1
            cnode=pnode(newloc, wlist, cost)

    return cnode.getacts()
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
dstar = dStarLiteSearch
