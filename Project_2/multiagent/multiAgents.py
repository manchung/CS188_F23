# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, os

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        super()
        self.travelled = util.Counter()

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def closestFoodDistance(self, pos, foodPos):
        currMin = 10000
        for food in foodPos.asList():
            currMin = min(manhattanDistance(pos, food), currMin)
        return currMin
    
    def sumFoodDistance(self, pos, foodPos):
        total = 0
        for food in foodPos.asList():
            total += manhattanDistance(pos, food)
        return total

    def freeSquares(self, pos, walls):
        total = 0
        x, y = pos
        for p, q in [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1), (x+1,y+1)]:
            if not walls[p][q]:
                total += 1
        return total


    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldPos = currentGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = currentGameState.getWalls()

        self.travelled[oldPos] += 1
        debug = os.getenv('debug')
        # print(f"debug: {debug}")
        "*** YOUR CODE HERE ***"
        # print(f"newFood: {newFood.asList()}")
        # print(f"Curr Pos: {currentGameState.getPacmanPosition()}")
        # print(f"New Pos: {newPos}")
        # print(f"Number of old food: {oldFood.count()}")
        # print(f"Number of new food: {newFood.count()}")
        # print(f"action: {action}")
        # print(f"Walls: {walls.asList()}")
        # input("Continue?")
        if debug == "1":
            print(f"{currentGameState.getPacmanPosition()} -> {newPos}")

        # return successorGameState.getScore()
        override = 0
        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                override = -100
        if newPos == oldPos: 
            override = -10
        
        # score += 1 / (1 + self.sumFoodDistance(newPos, newFood))
        newPosToClosestFood = self.closestFoodDistance(newPos, newFood)
        oldPosToClosestFood = self.closestFoodDistance(oldPos, newFood)
        newPosToAllFood = self.sumFoodDistance(newPos, newFood)
        oldPosToAllFood = self.sumFoodDistance(oldPos, newFood)

        if debug == "1":
            print(f"newPosToClosestFood: {newPosToClosestFood}")
            # print(f"oldPosToClosestFood: {oldPosToClosestFood}")
            print(f"newPosToAllFood: {newPosToAllFood}")
            # print(f"oldPosToAllFood: {oldPosToAllFood}")
            print(f"food count : {newFood.count()}")
            print(f"freeSquares: {self.freeSquares(newPos, walls)}")
            # input("Continue?")
        
        # if newPosToClosestFood != oldPosToClosestFood:
        #     score += 1 / (1 + newPosToClosestFood)
        # # elif newPosToAllFood != oldPosToAllFood:
        # #     score += 1 / (1 + newPosToAllFood)
        # else:
        #     print("same")
        score = 0
        score += 2 / (1 + newPosToClosestFood)
        score += 1000 / (1 + newFood.count())

        # if action == 'North':
        #     score += 0.0001
        # elif action == 'East':
        #     score += 0.0002
        
        score += 0.0001 * self.freeSquares(newPos, walls)
        # return 1 / (1 + self.closestFoodDistance(newPos, newFood)) + 10 / (1 + newFood.count())
        score_vec = (override, -self.travelled[newPos], -newFood.count(), -newPosToClosestFood, 
                     -newPosToAllFood, self.freeSquares(newPos, walls))
        if debug == "1":
            print(f"score: {score_vec}")
        return score_vec
        # return 1 / (1 + self.sumFoodDistance(newPos, newFood)) + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def optimizeActionHelper(self, gameState, agentIndex, depthBudget):
        # print(f"agentIndex: {agentIndex}  depthBudget: {depthBudget}")
        # if depthBudget == 0:
        #     print(f"self.evaluationFunction(gameState): {self.evaluationFunction(gameState)}")
        #     return (self.evaluationFunction(gameState), None)
        
        if agentIndex == 0:
            comp = lambda x, y: x > y
        else:
            comp = lambda x, y: x < y
        
        nextAgent = (agentIndex+1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepthBudget = depthBudget - 1
        else:
            nextDepthBudget = depthBudget
        
        actions = gameState.getLegalActions(agentIndex)
        bestAction = None
        bestScore = None
        # print(f"Number of actions: {len(actions)}")
        assert len(actions) > 0
        for action in actions:
            # print(f"action: {action}")
            successorGame = gameState.generateSuccessor(agentIndex, action)
            if successorGame.isWin() or successorGame.isLose() or nextDepthBudget == 0:
                score = self.evaluationFunction(successorGame)
            # if (agentIndex == 0 and successorGame.isWin()) or (agentIndex != 0 and successorGame.isLose()):
            #     return (self.evaluationFunction(successorGame), action)
            else:
                score, _ = self.optimizeActionHelper(successorGame, nextAgent, nextDepthBudget)
                # print(f"score for nextAgent {nextAgent} = {score}")
            
            if bestScore is None or comp(score, bestScore):
                bestScore = score
                bestAction = action
        assert bestAction is not None
        return (bestScore, bestAction)
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        bestScore, bestAction = self.optimizeActionHelper(gameState, 0, self.depth)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def optimizeActionHelper(self, gameState, agentIndex, depthBudget, alpha=-9999, beta=9999):
        # print(f"agentIndex: {agentIndex}  depthBudget: {depthBudget}")
        # if depthBudget == 0:
        #     print(f"self.evaluationFunction(gameState): {self.evaluationFunction(gameState)}")
        #     return (self.evaluationFunction(gameState), None)
        
        if agentIndex == 0:
            comp = lambda x, y: x > y
        else:
            comp = lambda x, y: x < y
        
        nextAgent = (agentIndex+1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepthBudget = depthBudget - 1
        else:
            nextDepthBudget = depthBudget
        
        actions = gameState.getLegalActions(agentIndex)
        bestAction = None
        bestScore = None
        # print(f"Number of actions: {len(actions)}")
        assert len(actions) > 0
        for action in actions:
            # print(f"action: {action}")
            successorGame = gameState.generateSuccessor(agentIndex, action)
            if successorGame.isWin() or successorGame.isLose() or nextDepthBudget == 0:
                score = self.evaluationFunction(successorGame)
            # if (agentIndex == 0 and successorGame.isWin()) or (agentIndex != 0 and successorGame.isLose()):
            #     return (self.evaluationFunction(successorGame), action)
            else:
                score, _ = self.optimizeActionHelper(successorGame, nextAgent, nextDepthBudget, alpha, beta)
                # print(f"score for nextAgent {nextAgent} = {score}")
            
            if bestScore is None or comp(score, bestScore):
                bestScore = score
                bestAction = action
            
            if agentIndex == 0 and comp(bestScore, beta) or agentIndex > 0 and comp(bestScore, alpha):
                return (bestScore, bestAction)
            
            if agentIndex == 0 and comp(score, alpha):
                alpha = score
            if agentIndex > 0 and comp(score, beta):
                beta = score
        
        assert bestAction is not None
        return (bestScore, bestAction)
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        bestScore, bestAction = self.optimizeActionHelper(gameState, 0, self.depth)
        return bestAction

import numpy as np 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectedUtility(self, gameState, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        num_actions = len(actions)
        assert num_actions > 0

        exp_val = 0
        for action in actions:
            successorGame = gameState.generateSuccessor(agentIndex, action)
            score = self.evaluationFunction(successorGame)
            exp_val += score
        
        return exp_val / num_actions


    def optimizeActionHelper(self, gameState, agentIndex, depthBudget):
        # print(f"agentIndex: {agentIndex}  depthBudget: {depthBudget}")
        # if depthBudget == 0:
        #     print(f"self.evaluationFunction(gameState): {self.evaluationFunction(gameState)}")
        #     return (self.evaluationFunction(gameState), None)
        
        # if agentIndex == 0:
        #     comp = lambda x, y: x > y
        # else:
        #     comp = lambda x, y: x < y
        
        nextAgent = (agentIndex+1) % gameState.getNumAgents()
        if nextAgent == 0:
            nextDepthBudget = depthBudget - 1
        else:
            nextDepthBudget = depthBudget
        
        actions = gameState.getLegalActions(agentIndex)
        # bestAction = None
        # bestScore = None
        actionScores = []
        # print(f"Number of actions: {len(actions)}")
        assert len(actions) > 0
        for action in actions:
            # print(f"action: {action}")
            successorGame = gameState.generateSuccessor(agentIndex, action)
            if successorGame.isWin() or successorGame.isLose() or nextDepthBudget == 0:
                score = self.evaluationFunction(successorGame)
            # if (agentIndex == 0 and successorGame.isWin()) or (agentIndex != 0 and successorGame.isLose()):
            #     return (self.evaluationFunction(successorGame), action)
            else:
                # score = self.expectedUtility(successorGame, nextAgent)
                score, _ = self.optimizeActionHelper(successorGame, nextAgent, nextDepthBudget)
                # print(f"score for nextAgent {nextAgent} = {score}")
            
            # if bestScore is None or comp(score, bestScore):
            #     bestScore = score
            #     bestAction = action
            actionScores.append((score, action))
        assert len(actionScores) > 0

        if agentIndex == 0:
            return max_vec(actionScores, key=lambda x: x[0])
        else:
            # avg = sum([k[0] for k in actionScores]) / len(actionScores)
            mean = avg_vec([k[0] for k in actionScores])
            return (mean, None)
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        bestScore, bestAction = self.optimizeActionHelper(gameState, 0, self.depth)
        return bestAction

def closestFoodDistance(pos, foodPos, capsules=[]):
    currMin = 10000
    for food in foodPos.asList() + capsules:
        currMin = min(manhattanDistance(pos, food), currMin)
    return currMin

def sumFoodDistance(pos, foodPos, capsules=[]):
    total = 0
    for food in foodPos.asList() + capsules:
        total += manhattanDistance(pos, food)
    return total

def freeSquares(pos, walls):
    total = 0
    x, y = pos
    for p, q in [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x-1,y-1), (x-1,y+1), (x+1,y-1), (x+1,y+1)]:
        if not walls[p][q]:
            total += 1
    return total

def avg_vec(vals):
    assert len(vals) > 0
    if isinstance(vals[0], tuple) or isinstance(vals[0], list):
        return list(np.average(vals, axis=0))
    else:
        return sum(vals)/len(vals)

def max_vec(vals, key):
    assert len(vals) > 0
    def comp(a, b):
        for i in range(min(len(a), len(b))):
            if a[i] > b[i]:
                return True
            elif b[i] > a[i]:
                return False
        return len(a) > len(b)

    if isinstance(key(vals[0]), tuple) or isinstance(key(vals[0]), list):
        m = vals[0]
        for v in vals[1:]:
            if comp(key(v), key(m)):
                m = v
        return m
    else:
        return max(vals, key=key)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    walls = currentGameState.getWalls()

    # self.travelled[oldPos] += 1
    debug = os.getenv('debug')
    # if food.count() < 2:
    #     return (scoreEvaluationFunction(currentGameState),)
    # return successorGameState.getScore()
    override = 0
    for ghost in ghostStates:
        if manhattanDistance(pos, ghost.getPosition()) <= 1:
            override = -100
    
    posToClosestFood = closestFoodDistance(pos, food, capsules)
    posToAllFood = sumFoodDistance(pos, food, capsules)
    
    # print(f"override: {override}, -foodCount: {-food.count()}, -closestFood: {-posToClosestFood}, -allFood: {-posToAllFood}, -capsules: {-len(capsules)}, freeSquares: {freeSquares(pos, walls)}")
    if food.count() < 2:
        # override = scoreEvaluationFunction(currentGameState)
        # score_vec = (override, 0, -food.count(), -posToClosestFood, 
        #                 -posToAllFood, -len(capsules), freeSquares(pos, walls))
        score_vec = (override, 0, -food.count() - len(capsules), -posToClosestFood, 
                        -posToAllFood, freeSquares(pos, walls))
    else:
        # score_vec = (override, 0, -food.count(), -posToClosestFood, 
        #                 -posToAllFood, -len(capsules), freeSquares(pos, walls))
        score_vec = (override, 0, -food.count() - len(capsules), -posToClosestFood, 
                        -posToAllFood, freeSquares(pos, walls))
        
    return score_vec

# Abbreviation
better = betterEvaluationFunction
# better = scoreEvaluationFunction
