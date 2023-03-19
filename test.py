import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use("TkAgg")


#
#   Grid Visualization Object
#
#   Create a figure for an M (rows) x N (column) grid.  The X-axis
#   will be the columns (to the right) and the Y-axis will be the rows
#   (top downward).
#
class Visualization:
    # Initialization.
    def __init__(self, states):
        # Save the dimensions and the states reference.
        self.rows   = len(states)
        self.cols   = len(states[0])
        self.states = states

        # Close the old figure, create a new figure/axes, turn off labels.
        plt.close()
        plt.figure()
        self.ax = plt.axes()
        self.ax.axis('off')

        # Draw the grid, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            self.ax.axhline(row, lw=1, color='b', zorder=1)
        for col in range(self.cols+1):
            self.ax.axvline(col, lw=1, color='b', zorder=1)

        # Add the external labels.
        for row in range(self.rows):
            self.write(row,        -1, str(row))
            self.write(row, self.cols, str(row))
        for col in range(self.cols):
            self.write(       -1, col, str(col))
            self.write(self.rows, col, str(col))

        # Create the colored boxes.
        self.boxes = self.colorboxes()

        # Force the figure to pop up.
        plt.pause(0.001)

    # Set the colored boxes according to the states.
    def colorboxes(self):
        # Determine the colors.
        color = np.ones((self.rows,self.cols,3))
        for row in range(self.rows):
            for col in range(self.cols):
                color[row,col,:] = self.states[row][col].color()

        # Set the boxes.
        return self.ax.imshow(color, interpolation='none', aspect='equal',
                              extent=[0, self.cols, 0, self.rows], zorder=0)

    # Add some text - this won't show until the next pause().
    def write(self, row, col, text):
        plt.text(0.33 + col, self.rows - 0.67 - row, text)

    # Update, changing the box colors according to the states.
    def update(self):
        # Remove the previous boxes and replace with new colors.
        self.boxes.remove()
        self.boxes = self.colorboxes()

        # Force the figure to update.  And wait to hit enter.
        plt.pause(0.001)
        input('Hit return to continue')


#
#   State Object
#
#   The state object (one per box/element in the grid), includes the
#   status (UNKNOWN, ONDECK, PROCESSED), cost, as well as a list of
#   neighbors.
#
#   Note we also allow a status of WALL to help the visualization.  As
#   well as PATH to mark the final path.
#
class State:
    # Possible status of each state.
    WALL      = -1      # Not a legal state - just to indicate the wall
    UNKNOWN   =  0      # "Air"
    ONDECK    =  1      # "Leaf"
    PROCESSED =  2      # "Trunk"
    PATH      =  3      # Processed and later marked as on path to goal

    STATUSSTRING = {WALL:      'WALL',
                    UNKNOWN:   'UNKNOWN',
                    ONDECK:    'ONDECK',
                    PROCESSED: 'PROCESSED',
                    PATH:      'PATH'}

    STATUSCOLOR = {WALL:      np.array([0.0, 0.0, 0.0]),   # Black
                   UNKNOWN:   np.array([1.0, 1.0, 1.0]),   # White
                   ONDECK:    np.array([0.0, 1.0, 0.0]),   # Green
                   PROCESSED: np.array([0.0, 0.0, 1.0]),   # Blue
                   PATH:      np.array([1.0, 0.0, 0.0])}   # Red

    # Initialization
    def __init__(self, row, col):
        # Save the location.
        self.row = row
        self.col = col

        # Clear the status and costs.
        self.status = State.UNKNOWN
        #self.creach = 0.0       # Actual cost to reach
        #self.cost   = 0.0       # Estimated total path cost (to sort)
        self.rhs = np.inf
        self.g = np.inf
        # Clear the references.
        self.parent    = None
        self.neighbors = []
        self.goal = self

    # Define less-than, so we can sort the states by cost.
    def __lt__(self, other):
        if min(self.g,self.rhs) +self.distance(self.goal) < min(other.g,other.rhs) +other.distance(other.goal):
            return True
        elif min(self.g,self.rhs) +self.distance(self.goal) == min(other.g,other.rhs) +other.distance(other.goal):
            if min(self.g,self.rhs) < min(other.g,other.rhs):
                return True
        else:
            return False
        #return self.cost < other.cost

    # Define the Manhattan distance.
    def distance(self, other):
        if self.status != State.WALL and other.status != State.WALL:
            return abs(self.row - other.row) + abs(self.col - other.col)
        else:
            return np.inf


    # Return the color matching the status.
    def color(self):
        return State.STATUSCOLOR[self.status]

    # Return the representation.
    def __repr__(self):
        return ("<State %d,%d = %s, g %f,rhs %f,k1 %f,k2 %f>\n" %
                (self.row, self.col,
                 State.STATUSSTRING[self.status],self.g,self.rhs, min(self.g,self.rhs) +self.distance(self.goal),min(self.g,self.rhs)))



#
#   A* Algorithm
#
# Estimate the cost to go from state to goal.
def costtogo(state, goal):
    return  1 * state.distance(goal)

def updateVertex(start,node,U):
    #print("updating Vertex")
    #print(node)
    if node != start:
        for neighbor in node.neighbors:
            tmp = node.rhs
            #print(neighbor)
            #print(neighbor.distance(node))
            node.rhs = min(node.rhs, neighbor.g + neighbor.distance(node))
            #print(node.g)
            #print(node.rhs)
            #print(tmp)
            if tmp != node.rhs:
                if(neighbor.parent != node):
                    #print("change connection from")
                    #print(node.parent)
                    #print("to")
                    #print(neighbor)
                    node.parent = neighbor
                    #print(neighbor)

    if node in U:
        U.remove(node)
        #print("remove")
        #print(node)
    if node.g != node.rhs:
        bisect.insort(U, node)
        #print("insert")
        #print(node)
    #print(node.g)
    #print(node.rhs)
def calculateKey(node,goal):
    return [min(node.g,node.rhs) +node.distance(goal) ,min(node.g,node.rhs)]

def compareKey(node1,node2,goal):
    key1 = calculateKey(node1,goal)
    key2 = calculateKey(node2,goal)
    if key1[0] < key2[0]:
        return True
    elif key1[0] == key2[0]:
        if key1[1] < key2[1]:
            return True
    else:
        return False
def computeShortestPath(U,start,goal):

        #U.sort()
        while (goal.g != goal.rhs or compareKey(U[0],goal,goal)):
            #print("U")
            #print(U)
            #U.sort()
            #print("computing")
            #print(U[0])
            node = U.pop(0)
            if (node.g > node.rhs):
                node.g = node.rhs

            else:
                node.g = np.inf
                updateVertex(start, node, U)
            for neighbor in node.neighbors:
                updateVertex(start, neighbor, U)
            #print("computing done")
            #print(goal)
            #if len(U) == 0:
            #    break
        #print("iteration done")

#def findChange(node,start,goal):
#    while(goal.parent != start):
#        if goal.parent == node:
def getPath(goal):
    path = []
    while goal.parent != None:
        goal.status = State.PATH
        path.append(goal)
        goal = goal.parent
    goal.status = State.PATH
    path.append(goal)
    path.reverse()
    #print("path")
    #print(path)

def resetPathVisual(goal):
    path = []
    while goal.parent != None:
        goal.status = State.UNKNOWN
        path.append(goal)
        goal = goal.parent
    goal.status = State.UNKNOWN



def updateObstacle(updateList,row,col,states):
    states[row][col].status = State.WALL
    states[row][col].rhs = np.inf
    states[row][col].g = np.inf
    for node in findDirectChild(states, states[row][col]):
        updateList.append(node)
    return updateList
def removeObstacle(updateList,row,col,states):
    states[row][col].status = State.UNKNOWN
    states[row][col].rhs = 100
    states[row][col].g = np.inf
    states[row][col].neighbors = [states[row-1][col],states[row+1][col],states[row][col-1],states[row][col+1]]
    for node in findDirectChild(states, states[row][col]):
        updateList.append(node)
    return updateList

def findDirectChild(states,node):
    Dlist = []
    row = node.row
    col = node.col
    if row - 1 >= 0:
        if states[row-1][col].parent == states[row][col]:
            Dlist.append(states[row-1][col])
    if row + 1 < len(states):
        if states[row+1][col].parent == states[row][col]:
            Dlist.append(states[row+1][col])
    if col - 1 >=0:
        if states[row][col-1].parent == states[row][col]:
            Dlist.append(states[row][col-1])
    if col + 1 < len(states[0]):
        if states[row][col+1].parent == states[row][col]:
            Dlist.append(states[row][col+1])
    return Dlist

def updateCost(states,goal,row,col,U,start,change):
    print(row,col)
    #updateVertex(start,states[row][col],U)
    if change == 1:
        states[row][col].rhs=np.inf
        states[row][col].g = np.inf
    elif change == -1:
        states[row][col].rhs = 100
        states[row][col].g = np.inf

    if row - 1 >= 0:
        if states[row-1][col].parent == states[row][col]:
            #states[row-1][col].rhs = np.inf
            #states[row-1][col].parent = None
            updateCost(states,goal,row-1,col,U,start,change)
    if row + 1 < len(states):
        if states[row+1][col].parent == states[row][col]:
            #states[row+1][col].rhs = np.inf
            #states[row +1][col].parent = None
            updateCost(states,goal,row+1,col,U,start,change)
    if col - 1 >=0:
        if states[row][col-1].parent == states[row][col]:
            #states[row][col-1].rhs = np.inf
            #states[row][col-1].parent = None
            updateCost(states,goal,row,col-1,U,start,change)
    if col + 1 < len(states[0]):
        if states[row][col+1].parent == states[row][col]:
            #states[row][col+1].rhs = np.inf
            #states[row][col+1].parent = None
            updateCost(states,goal,row,col+1,U,start,change)


# Run the full A* algorithm.
#def LPastar(start, goal, visual):
#    # Prepare the still empty *sorted* on-deck queue.
#    U= []
#    start.rhs = 0
#    bisect.insort(U, start)

    # Continually expand/build the search tree.
#    print("Starting the processing...")
#    while True:
#        # Show the grid.
#        print(U)
#        #visual.update()
#        computeShortestPath(U,start,goal)
#        print(goal.parent)
#        sig = input("What to do?")
#        if sig == "1":
#             break
#         if sig == "2":
#             states[2][7].status = State.WALL
#
#     # Show the final grid.
#     print("Goal state has been processed.")
#     #visual.update()
#     print(goal.parent)
#     # Create the path to the goal (backwards) and show.
#     print("Marking path...")
#     #############
#     path = []
#     states[2][11].parent = states[2][10]
#     while goal.parent != None:
#         goal.status = State.PATH
#         path.append(goal)
#         goal = goal.parent
#     goal.status = State.PATH
#     path.append(goal)
#     path.reverse()
#     print("path")
#     print(path)
#     #############
#     visual.update()
#     visual.update()
#     return



#
#   Main Code
#
def main():
    # Create a grid of states.
    M = 20
    N = 20
    addtotal = 0
    removetotal = 0
    for (r, c) in [(1,1),(2,2)]:
        states = [[State(m,n) for n in range(N)] for m in range(M)]

        # Define the walls.
        for m in range(M):
            states[m][0  ].status = State.WALL
            states[m][N-1].status = State.WALL
        for n in range(N):
            states[0  ][n].status = State.WALL
            states[M-1][n].status = State.WALL

        for (m,n) in [(3,4), (3,5), (3,6), (3,7), (3,8), (3,9),(2,7),(1,8),
                      (4,10), (5,11), (6,12), (7,13),
                      (7,7), (8,7), (9,7),(15,15),(14,15),(14,14),(15,16),(15,15),
                      (15,18),(15,17),(6,2),(6,3),(6,4),(8,2),(8,3),(8,4),(10,2),(10,3),(10,4),(7,2),(9,2),
                      (6,7),(6,8),(6,9),(8,7),(8,8),(8,9),(10,7),(10,8),(10,9),(7,7),(9,7),
                      (12,2),(12,3),(12,4),(14,2),(14,3),(14,4),(16,2),(16,3),(16,4),(13,4),(15,2),
                      (12,6),(12,7),(12,8),(14,6),(14,7),(14,8),(16,6),(16,7),(16,8),(13,8),(15,8),
                      (12,10),(13,10),(14,10),(12,12),(13,12),(14,12),(15,12),(16,12),(14,11)]:
        #for (n,m) in [(1,2),(1,3),(1,4),(1,5),(3,2),(3,3),(3,4),(3,5)]:
            states[m][n].status = State.WALL

        # Set the neighbors - this makes sure the full graph is implemented.
        for m in range(M):
            for n in range(N):
                if not states[m][n].status == State.WALL:
                    for (m1, n1) in [(m-1,n), (m+1,n), (m,n-1), (m,n+1)]:
                        #if not states[m1][n1].status == State.WALL:
                        states[m][n].neighbors.append(states[m1][n1])

        # Pick the start/goal states.
        #start = states[1][2]
        #goal  = states[6][1]

        start = states[5][4]
        goal = states[18][7]

        for sstate in states:
            for state in sstate:
                state.goal = goal
        # Create/update the visualization.
        visual = Visualization(states)
        visual.write(start.row, start.col, 'S')
        visual.write(goal.row,  goal.col,  'G')

        visual.update()
        visual.update()
        visual.update()
        # Run the A* algorithm.
        #LPastar(start, goal, visual)
        U = []
        start.rhs = 0
        bisect.insort(U, start)

        # Continually expand/build the search tree.
        print("Starting the processing...")
        while True:
            t1 = time.time()
            computeShortestPath(U, start, goal)
            t2 = time.time()
            runtime = t2 - t1
            print("time" + str(t2 - t1))
            updateList = []
            updateList = updateObstacle(updateList, r, c, states)
            updateCost(states, goal, r, c, U, start, 1)
            for node in updateList:
                updateVertex(start,node,U)
            t1 = time.time()
            computeShortestPath(U, start, goal)
            t2 = time.time()
            getPath(goal)
            visual.update()
            visual.update()
            visual.update()
            resetPathVisual(goal)
            visual.update()
            visual.update()
            visual.update()
            addtotal = addtotal + t2 - t1
            updateList = []
            updateList = removeObstacle(updateList, r, c, states)
            updateCost(states, goal, r, c, U, start, -1)
            updateVertex(start, states[r][c], U)
            for node in updateList:
                updateVertex(start, node, U)
            t1 = time.time()
            computeShortestPath(U, start, goal)
            t2 = time.time()
            getPath(goal)
            visual.update()
            visual.update()
            visual.update()
            resetPathVisual(goal)
            visual.update()
            visual.update()
            visual.update()
            removetotal = removetotal + t2 - t1
            break
    print(addtotal/2)
    print(removetotal/2)

    # Show the final grid.
    print("Goal state has been processed.")

    #visual.update()
    #visual.update()
    #visual.update()
    # Create the path to the goal (backwards) and show.
    print("Marking path...")
    #############
    path = []
    while goal.parent != None:
        goal.status = State.PATH
        path.append(goal)
        goal = goal.parent
    goal.status = State.PATH
    path.append(goal)
    path.reverse()
    print("path")
    print(path)
    #############
    visual.update()
    visual.update()



if __name__ == "__main__":
    states = main()
