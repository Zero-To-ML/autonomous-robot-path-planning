import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Choose start and goal coordinates that fall on the road network
start = np.array([121, 220])  # Coordinates in format [row, column]
goal = np.array([62, 203])  # Coordinates within the map boundaries

# Load the MIT map
# You'll need to replace this path with the correct location of your MIT map file
grid = np.load(r"D:\desktop\Autonomous robot path planning\mit.npy")

# Create copies of grid to be used for visualizing results
path = np.zeros([len(grid), len(grid[0])], dtype=float)
path -= 1000  # Initialize with negative values to distinguish unexplored areas
best_path = np.zeros([len(grid), len(grid[0])], dtype=int)

class BFSSearch:
    def __init__(self, start, goal, grid, path):
        self.start = start
        self.goal = goal
        self.goal_str = str(goal)
        self.grid = grid
        self.path = path
        self.queue = deque([start])  # Queue for BFS
        self.explored = {}  # Dictionary to keep track of explored nodes
        self.parent = {}  # Dictionary to store the parent of each node for path reconstruction
        self.explored[str(start)] = 0
        self.parent[str(start)] = None
        
    def run_bfs(self):
        """Run the BFS algorithm until the goal is found or the queue is empty"""
        explored_count = 0
        
        while self.queue:
            current_pos = self.queue.popleft()
            current_pos_str = str(current_pos)
            current_depth = self.explored[current_pos_str]
            
            # Mark on visualization path
            self.path[current_pos[0], current_pos[1]] = current_depth
            
            # Check if we've reached the goal
            if np.array_equal(current_pos, self.goal):
                print("Goal found!")
                return True
                
            # Get possible moves from current position
            possible_moves = self.generate_potential_moves(current_pos)
            
            # Process each possible move
            for move in possible_moves:
                move_str = str(move)
                if self.valid_move(move) and move_str not in self.explored:
                    self.queue.append(move)
                    self.explored[move_str] = current_depth + 1
                    self.parent[move_str] = current_pos_str
            
            # Print progress indicator
            if explored_count % 1000 == 0:
                print(f"Explored Count: {explored_count}")
            explored_count += 1
        
        print("No path found!")
        return False
    
    def reconstruct_path(self):
        """Reconstruct the path from start to goal using the parent dictionary"""
        if self.goal_str not in self.parent:
            print("Cannot reconstruct path: goal not reached")
            return False
            
        current = self.goal
        goal_count = 0
        
        while not np.array_equal(current, self.start):
            best_path[current[0], current[1]] = 1
            goal_count += 1
            
            # Get parent
            parent_str = self.parent[str(current)]
            current = self.string_to_array(parent_str)
            
            # Safety check
            if goal_count > 1000:
                print("Path reconstruction stopped: exceeded maximum path length")
                break
                
        # Mark start position
        best_path[self.start[0], self.start[1]] = 1
        print(f"Moves To Goal: {goal_count}")
        return True
            
    # Helper Functions
    def generate_potential_moves(self, pos):
        """Generate potential moves from the current position (8-connected)"""
        u = np.array([-1, 0])   # up
        d = np.array([1, 0])    # down
        l = np.array([0, -1])   # left
        r = np.array([0, 1])    # right
        
        ul = np.array([-1, -1]) # up-left
        ur = np.array([-1, 1])  # up-right
        dl = np.array([1, -1])  # down-left
        dr = np.array([1, 1])   # down-right

        # Using 8-connectivity (including diagonals)
        potential_moves = [pos + u, pos + d, pos + l, pos + r, 
                          pos + ul, pos + ur, pos + dl, pos + dr]
        return potential_moves

    def valid_move(self, move):
        """Check if a move is valid (within map boundaries and on a road)"""
        # Check if out of boundary
        if (move[0] < 0) or (move[0] >= len(self.grid)):
            return False
        if (move[1] < 0) or (move[1] >= len(self.grid[0])):
            return False
        # Check if path exists (value >= 0.5 indicates road)
        if self.grid[move[0], move[1]] < 0.5:
            return False
        return True
        
    def string_to_array(self, string):
        """Convert a string representation of coordinates back to a numpy array"""
        string = string.replace('[', '')
        string = string.replace(']', '')
        string = string.split()
        array = [int(string[0]), int(string[1])]
        return np.array(array)

# Initialize BFS
bfs = BFSSearch(start, goal, grid, path)

# Run BFS algorithm
bfs.run_bfs()

print('')
print(f'Fully explored count: {len(path[path > 0])}')

# First visualization - BFS exploration with start/goal markers and all roads
plt.figure(figsize=(10, 6))

# Create a base layer with all roads visible (gray)
roads_layer = np.zeros_like(grid)
roads_layer[grid > 0.5] = 0.3  # Set all road pixels to a light gray value

# Show the base map with all roads
plt.imshow(roads_layer, cmap='Greys', alpha=1.0)

# Overlay the explored paths with a colormap showing distance
plt.imshow(path, cmap='jet', alpha=0.7)

# Add start and goal markers
plt.plot(start[1], start[0], 'go', markersize=12, label='Start')  # Green circle for start
plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')     # Red circle for goal

plt.title('BFS Search Exploration')
plt.legend(loc='upper right')
plt.colorbar(label='Distance from Start')
plt.tight_layout()
plt.show()

# Reconstruct the path
bfs.reconstruct_path()

# Second visualization - Final path with start/goal markers and all roads
plt.figure(figsize=(10, 6))

# Create a base layer with all roads visible (gray)
roads_layer = np.zeros_like(grid)
roads_layer[grid > 0.5] = 0.3  # Set all road pixels to a light gray value

# Show the base map with all roads
plt.imshow(roads_layer, cmap='Greys', alpha=1.0)

# Overlay the best path
plt.imshow(best_path, cmap='viridis', alpha=0.9)

# Add start and goal markers
plt.plot(start[1], start[0], 'go', markersize=12, label='Start')  # Green circle for start
plt.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')     # Red circle for goal

plt.title('BFS Search Optimal Path')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()