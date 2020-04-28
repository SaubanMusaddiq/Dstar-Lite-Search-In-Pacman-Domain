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

import util, time, copy

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
    cstate = problem.getStartState()
    nlist = []
    cnode = pnode(cstate, nlist, 9999) 
    actstack = util.Stack()
    sStack = util.Stack()	
    Set = set()
    sStack.push(cnode)
    Set.add(problem.getStartState())
    if problem.isGoalState(problem.getStartState()):
        wact= []
        return wact
    while not problem.isGoalState(cnode.getstate()):
       # if len(problem.getSuccessors(cnode.getstate())) == 0:
        #    if sStack.isEmpty():
         #       return
          #  cnode = sStack.pop()
        test = 0
        for wx in problem.getSuccessors(cnode.getstate()):
            if wx not in Set:
                wlist = cnode.getacts()[:]
                wlist.append(wx[1])
                wnode = pnode(wx[0], wlist, 9999)
                sStack.push(wnode)
        if sStack.isEmpty():
            return

        while(not sStack.isEmpty()):
            mstate = sStack.pop()
            if mstate.getstate() not in Set:       
                Set.add(mstate.getstate())
                cnode = mstate
                if problem.isGoalState(cnode.getstate()):
                    wact = cnode.getacts()
                    return wact
                break
 #       if test == 0 and cstate == problem.getStartState():
  #          return                
   #     elif test == 0:
    #        print("previous state failed: ")
     #       print(cstate)
      #      cstate = sStack.pop()
       #     print("popping: new state is")
        #    print(cstate)
         #   if not actstack.isEmpty():
    #            actstack.pop()
    

    util.raiseNotDefined()



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
  #  Set = set()
    cstate = problem.getStartState()
    slist = []
    wlist = []
    slist.append(problem.getStartState())
    snode = pnode(cstate, wlist, 9999)
    cnode = snode
    #Set.add(problem.getStartState()[0])
    sQueue = util.Queue() 
    while not problem.isGoalState(cnode.getstate()):
        for successor in problem.getSuccessors(cnode.getstate()):
#            if successor[0] not in Set:
            if successor not in slist:
                wlist = cnode.getacts()[:]
                wlist.append(successor[1])
                newnode = pnode(successor[0], wlist, 9999)
                sQueue.push(newnode)
        test = 0
        while not sQueue.isEmpty():
            cnode = sQueue.pop()
#            if cnode.getstate()[0] not in Set:
            if cnode.getstate() not in slist:
                test = 1
     #           Set.add(cnode.getstate())
                slist.append(cnode.getstate())
                break
        if test == 0:
            if problem.isGoalState(cnode.getstate()):
                return cnode.getacts()
            return
    return cnode.getacts()
    
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    Set = set()
    cstate = problem.getStartState()
    Set.add(cstate)
    slist = []
    snode = pnode(cstate, slist, 0)
    cnode = snode
    sQueue = util.PriorityQueue() 
    while not problem.isGoalState(cstate):
        for successor in problem.getSuccessors(cnode.getstate()):
            if successor not in Set:
                wlist = cnode.getacts()[:]
                wlist.append(successor[1])
                newnode = pnode(successor[0], wlist, successor[2]+cnode.getcost())
                sQueue.push(newnode, successor[2]+cnode.getcost())
        test = 0
        while not sQueue.isEmpty():
            cnode = sQueue.pop()
            if cnode.getstate() not in Set:
                test = 1
                Set.add(cnode.getstate())
                cstate = cnode.getstate()
                break
        if test == 0:
            return
    return cnode.getacts()



    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    

#    Set = set()
    cstate = problem.getStartState()
#    Set.add(cstate[0])
    tlist = []
    slist = []
    snode = pnode(cstate, tlist, 0)
    slist.append(problem.getStartState())
    cnode = snode
    hval=heuristic(cstate, problem)
    sQueue = util.PriorityQueue() 
    print(problem.startState)
    print(problem.goal)
    while not problem.isGoalState(cstate):
        for successor in problem.getSuccessors(cnode.getstate()):
#            if successor[0] not in Set:
            if successor not in slist:
                wlist = cnode.getacts()[:]
                wlist.append(successor[1])
                newnode = pnode(successor[0], wlist, successor[2]+cnode.getcost())
                hval = heuristic(successor[0], problem)
                sQueue.push(newnode, successor[2]+cnode.getcost()+hval)
        test = 0
        while not sQueue.isEmpty():
            cnode = sQueue.pop()
           # print("pop")
           # print(cnode.getstate())
#            if cnode.getstate() not in Set:
            if cnode.getstate() not in slist:
           #     print("cost is:")
           #     print(cnode.getcost())
                test = 1
#                Set.add(cnode.getstate())
                slist.append(cnode.getstate())
                cstate = cnode.getstate()
                break
        if test == 0:
            if problem.isGoalState(cnode.getstate()):
                return cnode.getacts()
            return
    return cnode.getacts()


    util.raiseNotDefined()

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
        


def astar2(problem, view_range=2):
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
        for i in range(max(1,pos[0]-view_range),min(walls.width,pos[0]+view_range)):
            for j in range(max(1,pos[1]-view_range),min(walls.height,pos[1]+view_range)):
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


#    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
