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
from functools import partial


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
    return path


def lpaStar(problem, heuristic=nullHeuristic):
    lpastar = lpaStarSearch(problem)
    path = lpastar.computePath()
    return path


class lpaStarSearch(object):
    def __init__(self, problem, visibility=2):
        self.prob = problem
        self.visibility = visibility
        self.state = problem.getStartState()
        self.goal = problem.getGoalState()
        self.g_map = {}
        self.rhs_map = {}
        self.rhs_map[self.state] = 0
        # self.key_modifier = 0
        self.frontier = util.PriorityQueue()
        self.frontier.push(self.state, self.buildKeyTuple(self.goal))

    def consistentNode(self, node):
        return self.rhsValue(self.state) == self.gValue(self.state)

    def rhsValue(self, node):
        return 0 if self.prob.isGoalState(node) else self.rhs_map.get(node, float('inf'))

    def gValue(self, node):
        return self.g_map.get(node, float('inf'))

    def heuristic(self, xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def computeRhsValue(self, node):
        return self.lookaheadCost(node, self.minCostSuccessor(node))

    def lookaheadCost(self, node, successor):
        succ,_,cost = successor
        return self.gValue(succ) + cost

    def minCostSuccessor(self, node):
        cost = partial(self.lookaheadCost, node)
        return min(self.prob.getSuccessors(node), key=cost)

    def buildKeyTuple(self, node):
        min_g_rhs = min([self.gValue(node), self.rhsValue(node)])
        return (min_g_rhs + self.heuristic(node, self.goal), min_g_rhs)

    def updateNodes(self, nodes):
        # for node in nodes:
        #     if not self.prob.isGoalState(node):
        #         self.rhs_map[node] = self.computeRhsValue(node)
        #     if self.gValue(node) != self.rhsValue(node):
        #         self.frontier.update(node, self.buildKeyTuple(node))
        # for node in nodes:
        #     if node != self.state:
        #         self.rhs_map[node] = self.computeRhsValue(node)
        #     if self.gValue(node) != self.rhsValue(node): # Insert & Remove?
        #         self.frontier.update(node, self.buildKeyTuple(node))
        print(nodes)
        for node in nodes:
            if (node != self.state):
                self.rhs_map[node] = self.minCostSuccessor(node);
                self.frontier.delete(node)
                if (self.gValue(node) != self.rhsValue(node)):
                    self.frontier.push(node, self.buildKeyTuple(node))


    def planPath(self):
        while not self.frontier.isEmpty() and (self.frontier.nsmallest(1)[0][0] < self.buildKeyTuple(self.goal) or not self.consistentNode(self.goal)): # change the startState to goal state
            # old_key = self.frontier.nsmallest(1)[0][0]
            node = self.frontier.pop()
            # new_key = self.buildKeyTuple(node)
            #     self.frontier.push(node, new_key)
            # if old_key < new_key:
            #     print(old_key, new_key)
            if self.gValue(node) > self.rhsValue(node): # OverConsistant
                self.g_map[node] = self.rhsValue(node)
                successors = [successor for successor,_,_ in self.prob.getSuccessors(node)]
                self.updateNodes(successors)
            else:                                          # UnderConsistant
                self.g_map[node] = float('inf')
                successors = [successor for successor,_,_ in self.prob.getSuccessors(node)]
                self.updateNodes(successors + [node])
            print(1)
            return
        return

    def computePath(self):
        path = []
        # changed_walls = self.prob.update_walls(self.state, self.visibility)
        # self.planPath()
        # last_starting_state = self.state

        while not self.prob.isGoalState(self.state): # forever

            # if self.gValue(self.state) == float('inf'):
            #     raise Exception("Blocked Path")
            self.planPath()
            print(self.state)
            self.state, action, cost = self.minCostSuccessor(self.state)
            print(action)

            changed_walls = self.prob.update_walls(self.state, self.visibility)
            path.append(action)
            print(changed_walls)
            if changed_walls:
                # self.key_modifier += self.heuristic(last_starting_state, self.state)
                # last_starting_state = self.state
                self.updateNodes({node for wall in changed_walls for node, action, cost in self.prob.getSuccessors(wall)})
                # self.planPath()
            break
        return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
lpastar = lpaStar
