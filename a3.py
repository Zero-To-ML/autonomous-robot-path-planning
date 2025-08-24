import numpy as np
import matplotlib.pyplot as plt

# Choose start and goal coordinates that fall on the road network (value = 1.0)
start = np.array([62, 203])  # Coordinates in format [row, column]
goal = np.array([121, 220])   # Updated to a valid point within the map boundaries

# Load the MIT map
grid = np.load(r"D:\desktop\Autonomous robot path planning\mit.npy")

# Convert 2D array to 3D to match the expected format in the valid_move function
grid_3d = np.zeros((grid.shape[0], grid.shape[1], 3))
for i in range(3):
    grid_3d[:,:,i] = grid

# Copies of grid to be used for visualizing results
path = np.zeros([len(grid), len(grid[0])], dtype=float)
path -= 1000
best_path = np.zeros([len(grid), len(grid[0])], dtype=int)


class AStarSearch:
    # [Class definition remains the same as before]
    def __init__(self, start, goal, grid, path):
        self.pos = start
        self.pos_str = str(start)
        self.pos_depth = 0
        self.goal_str = str(goal)
        self.explored = {}
        self.not_explored = {}
        self.not_explored[str(start)] = 0
        self.grid = grid
        self.path = path

    # START - Student Section
    def get_possible_moves(self):
        # Get Potential Moves
        possible_moves = self.generate_potential_moves(self.pos)

        # For each potential move:
        #   -Check if each potential move is valid.
        #   -Check if move has already been explored.
        #   -Add to not explored list if valid and not explored.
        for move in possible_moves:
            if self.valid_move(move):
                if (str(move) not in self.explored) and (str(move) not in self.not_explored):
                    self.not_explored[str(move)] = self.pos_depth + 1 + self.heuristic(move)

        # Since all next possible moves have been determined,
        # consider current location explored.
        self.explored[self.pos_str] = self.pos_depth
        return True

    def goal_found(self):
        if self.goal_str in self.not_explored:
            self.pos = self.string_to_array(self.goal_str)
            heurestic_cost = self.not_explored.pop(self.goal_str)
            self.path[self.pos[0], self.pos[1]] = heurestic_cost
            return True
        return False

    def explore_next_move(self):
        # Determine next move to explore.
        sorted_not_explored = sorted(self.not_explored,
                                    key=self.not_explored.get,
                                    reverse=False)

        # Determine the pos and depth of next move.
        self.pos_str = sorted_not_explored[0]
        self.pos = self.string_to_array(self.pos_str)
        self.pos_depth = round(self.not_explored.pop(self.pos_str) - self.heuristic(self.pos))
        
        # Write depth of next move onto path.
        self.path[self.pos[0], self.pos[1]] = round(self.pos_depth, 1)
        return True

    def heuristic(self, move):
        diff = move - self.string_to_array(self.goal_str)
        answer = np.sqrt(sum(diff**2))
        return round(answer, 1)

    # Helper Functions
    def generate_potential_moves(self, pos):
        u = np.array([-1, 0])
        d = np.array([1, 0])
        l = np.array([0, -1])
        r = np.array([0, 1])

        potential_moves = [pos + u, pos + d, pos + l, pos + r]
        # Include diagonal moves
        potential_moves += [pos + u+r, pos + u+l, pos + d+r, pos + d+l]
        return potential_moves

    def valid_move(self, move):
        # Check if out of boundary.
        if (move[0] < 0) or (move[0] >= len(grid)):
            return False
        if (move[1] < 0) or (move[1] >= len(grid[0])):
            return False
        # Check if wall or obstacle exists.
        # Modified to work with the binary MIT map format
        if self.grid[move[0], move[1], 0] < 0.5:  # Using 0.5 as threshold for binary values
            return False
        return True

    def string_to_array(self, string):
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.split()
        array = [int(string[0]), int(string[1])]
        return np.array(array)


# Init with the 3D grid
astar = AStarSearch(start, goal, grid_3d, path)

explored_count = 0
while True:
    # Determine next possible moves.
    astar.get_possible_moves()
    if astar.goal_found():
        break
    astar.explore_next_move()
    # Print Progress Indicator
    if explored_count % 1000 == 0:
        print("Explored Count: " + str(explored_count))
    explored_count += 1
    # Safety check to prevent infinite loops
    if len(astar.not_explored) == 0:
        print("No path found!")
        break

print('')
print('Fully explored count ' + str(len(path[path > 0])))

# First visualization - A* exploration with start/goal markers and all roads
plt.figure(figsize=(8, 4))

# Create a base layer with all roads visible (gray)
roads_layer = np.zeros_like(grid)
roads_layer[grid > 0.5] = 0.3  # Set all road pixels to a light gray value

# Show the base map with all roads
plt.imshow(roads_layer, cmap='Greys', alpha=1.0)

# Overlay the explored paths
plt.imshow(path, cmap='jet', alpha=0.75)

# Add start and goal markers
plt.plot(start[1], start[0], 'go', markersize=12, label='Start')  # Green circle for start
plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')     # Red circle for goal

plt.title('A* Search Exploration with All Available Paths')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

pos = goal
goal_count = 0
while True:
    best_path[pos[0], pos[1]] = 1
    h_pos = round(path[pos[0], pos[1]], 1)
    if h_pos == 1:
        break
    potential_moves = astar.generate_potential_moves(pos)
    for move in potential_moves:
        if not astar.valid_move(move):
            continue
        h_move = path[move[0], move[1]]
        if h_move == (h_pos - 1):
            goal_count += 1
            pos = move
            break
    # Safety check to prevent infinite loops
    if goal_count > 1000:
        print("Path reconstruction stopped: exceeded maximum path length")
        break

print('Moves To Goal: ' + str(goal_count))

# Second visualization - Final path with start/goal markers and all roads
plt.figure(figsize=(8, 4))

# Create a base layer with all roads visible (gray)
roads_layer = np.zeros_like(grid)
roads_layer[grid > 0.5] = 0.3  # Set all road pixels to a light gray value

# Show the base map with all roads
plt.imshow(roads_layer, cmap='Greys', alpha=1.0)

# Overlay the best path
plt.imshow(best_path, cmap='jet', alpha=0.75)

# Add start and goal markers
plt.plot(start[1], start[0], 'go', markersize=12, label='Start')  # Green circle for start
plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')     # Red circle for goal

plt.title('A* Search Path with All Available Paths')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()