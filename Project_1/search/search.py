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
from game import Directions
s = Directions.SOUTH
w = Directions.WEST

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

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # each p_queue item is a tuple (state, action_list, cum_cost) where action_list
    # is the action sequence from start to that state, and cum_cost is the 
    # cumulative cost so far from start to that state.
    p_queue = util.PriorityQueue()

    # priority counter counts backward, to make sure the latest inserted state
    # has lowest priority in the queue
    p_counter = 0
    p_queue.push((problem.getStartState(), [], 0), p_counter)

    visited = set()
    while not p_queue.isEmpty():
        state, actions, cum_cost = p_queue.pop()
        if problem.isGoalState(state):
            return actions
        
        if state in visited:
            continue
        else:
            visited.add(state)

        # Not goal state. Expands state and insert each into queue
        for successor, action, step_cost in problem.getSuccessors(state):
            p_counter -= 1
            new_action_list = actions.copy()
            new_action_list.append(action)
            new_cum_cost = cum_cost + step_cost
            p_queue.push((successor, new_action_list, new_cum_cost), p_counter)

    # should not reach
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # each p_queue item is a tuple (state, action_list, cum_cost) where action_list
    # is the action sequence from start to that state, and cum_cost is the 
    # cumulative cost so far from start to that state.
    p_queue = util.PriorityQueue()

    # priority counter counts backward, to make sure the latest inserted state
    # has lowest priority in the queue
    p_counter = 0
    p_queue.push((problem.getStartState(), [], 0), p_counter)

    visited = set()
    while not p_queue.isEmpty():
        state, actions, cum_cost = p_queue.pop()
        # print(f'state: {state}, actions:{actions}')
        # input('Hit Enter')
        if problem.isGoalState(state):
            # print(f'Found solutions: {actions}')
            return actions
        
        if state in visited:
            continue
        else:
            visited.add(state)
        
        # Not goal state. Expands state and insert each into queue
        for successor, action, step_cost in problem.getSuccessors(state):
            p_counter += 1
            new_action_list = actions.copy()
            new_action_list.append(action)
            new_cum_cost = cum_cost + step_cost
            p_queue.push((successor, new_action_list, new_cum_cost), p_counter)

    # should not reach
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # each p_queue item is a tuple (state, action_list, cum_cost) where action_list
    # is the action sequence from start to that state, and cum_cost is the 
    # cumulative cost so far from start to that state.
    p_queue = util.PriorityQueue()

    # priority counter counts backward, to make sure the latest inserted state
    # has lowest priority in the queue
    
    startingState = problem.getStartState()
    p_queue.push((startingState, [], 0), 0)

    visited = set()
    while not p_queue.isEmpty():
        state, actions, cum_cost = p_queue.pop()
        if problem.isGoalState(state):
            return actions
        
        if state in visited:
            continue
        else:
            visited.add(state)
        
        # Not goal state. Expands state and insert each into queue
        for successor, action, step_cost in problem.getSuccessors(state):
            new_action_list = actions.copy()
            new_action_list.append(action)
            # new_cum_cost = cum_cost + step_cost
            new_cum_cost = problem.getCostOfActions(new_action_list)
            p_queue.push((successor, new_action_list, new_cum_cost), new_cum_cost)

    # should not reach
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
     # each p_queue item is a tuple (state, action_list, cum_cost) where action_list
    # is the action sequence from start to that state, and cum_cost is the 
    # cumulative cost so far from start to that state.
    p_queue = util.PriorityQueue()

    # priority counter counts backward, to make sure the latest inserted state
    # has lowest priority in the queue

    start_state = problem.getStartState()
    p_queue.push((start_state, [], 0), heuristic(start_state, problem))

    visited = set()
    while not p_queue.isEmpty():
        state, actions, cum_cost = p_queue.pop()
        if problem.isGoalState(state):
            return actions
        
        if state in visited:
            continue
        else:
            visited.add(state)
        
        # Not goal state. Expands state and insert each into queue
        for successor, action, step_cost in problem.getSuccessors(state):
            new_action_list = actions.copy()
            new_action_list.append(action)
            # new_cum_cost = cum_cost + step_cost
            new_cum_cost = problem.getCostOfActions(new_action_list)
            p_queue.push((successor, new_action_list, new_cum_cost), 
                         new_cum_cost + heuristic(successor, problem))

    # should not reach
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
