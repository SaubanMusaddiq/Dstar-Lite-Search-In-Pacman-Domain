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
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
