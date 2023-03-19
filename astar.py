#!/usr/bin/env python3
#
#   astar.py
#
#   HW1 A* Skeleton
#
#   This only shows the data structure and visualization.  Please
#   update to add/write the actual A* algorithm.
#
#   Note this defines a state object for each element/box in the grid.
#   Each state includes a list of neighbors.
#
#   It uses a 2D list of list of states to help the visualization.
#   But the actual A* algorithm may rely on the neighbor list and does
#   not need to consider (row/col) indices in the process.
#
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
        self.creach = 0.0       # Actual cost to reach
        self.cost   = 0.0       # Estimated total path cost (to sort)

        # Clear the references.
        self.parent    = None
        self.neighbors = []


    # Define less-than, so we can sort the states by cost.
    def __lt__(self, other):
        return self.cost < other.cost

    # Define the Manhattan distance.
    def distance(self, other):
        return abs(self.row - other.row) + abs(self.col - other.col)


    # Return the color matching the status.
    def color(self):
        return State.STATUSCOLOR[self.status]

    # Return the representation.
    def __repr__(self):
        return ("<State %d,%d = %s, cost %f>\n" %
                (self.row, self.col,
                 State.STATUSSTRING[self.status], self.cost))



#
#   A* Algorithm
#
# Estimate the cost to go from state to goal.
def costtogo(state, goal):
    return  1 * state.distance(goal)

# Run the full A* algorithm.
def astar(start, goal, visual):
    # Prepare the still empty *sorted* on-deck queue.
    t1 = time.time()
    onDeck = []
    Processed = []
    # Setup the start state/cost to initialize the algorithm.
    start.status = State.ONDECK
    start.creach = 0.0
    start.cost   = costtogo(start, goal)
    start.parent = None
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    #print("Starting the processing...")
    while True:
        # Show the grid.
        #visual.update()

        #############
        #print("onDeck")
        #print(onDeck)
        #print("processed")
        #print(Processed)
        Nlist = start.neighbors
        pnode = onDeck.pop(0)
        for node in Nlist:
            if(node.status == State.UNKNOWN):
                node.creach = start.creach + 1
                node.status = State.ONDECK
                node.cost = costtogo(node, goal) + node.creach
                node.parent = start
                bisect.insort(onDeck, node)
            elif(node.status == State.ONDECK):
                node.creach = start.creach + 1
                node.status = State.ONDECK
                node.cost = costtogo(node, goal) + node.creach
                node.parent = start
                onDeck.sort()

        pnode.status = State.PROCESSED

        start = onDeck[0]
        bisect.insort(Processed,pnode)
        if (goal.status == State.PROCESSED):
            break
        #############

    # Show the final grid.
    #print("Goal state has been processed.")
    #visual.update()

    # Create the path to the goal (backwards) and show.
    #print("Marking path...")
    #############
    t2 = time.time()
    print("in")
    print(t2-t1)
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
    #visual.update()
    #visual.update()
    return



#
#   Main Code
#
def main():
    # Create a grid of states.

    total = 0
    for (m, n) in [(5, 5)]:


        M = 20
        N = 20
        states = [[State(m, n) for n in range(N)] for m in range(M)]
        states[m][n].status = State.WALL
        # Define the walls.
        for m in range(M):
            states[m][0].status = State.WALL
            states[m][N - 1].status = State.WALL
        for n in range(N):
            states[0][n].status = State.WALL
            states[M - 1][n].status = State.WALL

        for (m, n) in [(3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (2, 7), (1, 8),
                       (4, 10), (5, 11), (6, 12), (7, 13),
                       (7, 7), (8, 7), (9, 7), (15, 15), (14, 15), (14, 14), (15, 16), (15, 15),
                       (15, 18), (15, 17), (6, 2), (6, 3), (6, 4), (8, 2), (8, 3), (8, 4), (10, 2), (10, 3), (10, 4),
                       (7, 2), (9, 2),
                       (6, 7), (6, 8), (6, 9), (8, 7), (8, 8), (8, 9), (10, 7), (10, 8), (10, 9), (7, 7), (9, 7),
                       (12, 2), (12, 3), (12, 4), (14, 2), (14, 3), (14, 4), (16, 2), (16, 3), (16, 4), (13, 4),
                       (15, 2),
                       (12, 6), (12, 7), (12, 8), (14, 6), (14, 7), (14, 8), (16, 6), (16, 7), (16, 8), (13, 8),
                       (15, 8),
                       (12, 10), (13, 10), (14, 10), (12, 12), (13, 12), (14, 12), (15, 12), (16, 12), (14, 11)]:
            states[m][n].status = State.WALL
    # Set the neighbors - this makes sure the full graph is implemented.
        for m in range(M):
            for n in range(N):
                if not states[m][n].status == State.WALL:
                    for (m1, n1) in [(m-1,n), (m+1,n), (m,n-1), (m,n+1)]:
                        if not states[m1][n1].status == State.WALL:
                            states[m][n].neighbors.append(states[m1][n1])

        # Pick the start/goal states.
        start = states[5][4]
        goal = states[18][7]

        # Create/update the visualization.
        visual = Visualization(states)
        #visual.write(start.row, start.col, 'S')
        #visual.write(goal.row,  goal.col,  'G')
        #visual.update()

        # Run the A* algorithm.
        t1 = time.time()
        astar(start, goal, visual)
        t2 = time.time()
        total = total + t2 - t1
    print("avg")
    print(total)
        # Report and return.
        #statuses  = [states[m][n].status for n in range(N) for m in range(M)]
        #unknown   = statuses.count(State.UNKNOWN)
        #ondeck    = statuses.count(State.ONDECK)
        #processed = statuses.count(State.PROCESSED) + statuses.count(State.PATH)

        #print("Solution cost %f" % goal.cost)
        #print("%d states processed" % processed)
        #print("%d states pending"   % ondeck)
        #print("%d states unreached" % unknown)

        #input("Hit return to end and close the figure")

if __name__ == "__main__":
    states = main()
