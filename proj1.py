# Import libaries

import math
import altair as alt
import pandas as pd

def write_output_file(nodes_path, start_coord, goal_coord, grid_dict, node_count, visited_count, file_path):
    # Initialize lists to store action indices and node costs
    actions_list, f_values, g_values, h_values = [], [], [], []

    # Start from the end of the path and work back to the start
    current = nodes_path[-1]
    while current.parent is not None:
        # Check each possible action to see which one leads to the current node
        for action, move in enumerate(actions, start=1):
            potential_parent_coord = (current.coord[0] - move[0], current.coord[1] - move[1])
            if current.parent.coord == potential_parent_coord:
                actions_list.insert(0, action)
                break
        # Insert the node's costs at the beginning of the lists
        f_values.insert(0, current.f)
        g_values.insert(0, current.g)
        h_values.insert(0, current.h)
        # Move to the parent of the current node
        current = current.parent

    # Add the costs for the start node
    f_values.insert(0, current.f)
    g_values.insert(0, current.g)
    h_values.insert(0, current.h)

    # Mark the path on the grid output
    grid_output = [[grid_dict.get((x, y), 1) for x in range(50)] for y in range(30)]
    for node in nodes_path:
        x, y = node.coord
        if (x, y) == start_coord:
            grid_output[y][x] = 2  # Start position
        elif (x, y) == goal_coord:
            grid_output[y][x] = 5  # Goal position
        else:
            grid_output[y][x] = 3  # Path

    # Write to the output file
    with open(file_path, 'w') as file:
        # Depth level of the goal node is the length of the path minus 1
        file.write(f"{len(nodes_path) - 1}\n")
        file.write(f"{node_count}\n")
        file.write(f"{visited_count}\n")
        file.write(' '.join(map(str, actions_list)) + '\n')
        file.write(' '.join(map(str, f_values)) + '\n')
        file.write(' '.join(map(str, g_values)) + '\n')
        file.write(' '.join(map(str, h_values)) + '\n')

        # Write the grid with the path marked
        for row in grid_output:
            file.write(' '.join(map(str, row)) + '\n')




# Reads in input file to store start coordinate, end coordinate, and the grid matrix
def read_input_file(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        start_x, start_y, goal_x, goal_y = map(int, first_line.split())
        grid_matrix = []
        for y in range(29, -1, -1):  
            line = file.readline().strip()
            if line: 
                cell_values = list(map(int, line.split()))
                grid_matrix.append([[x, y, cell_values[x]] for x in range(len(cell_values))])

    return (start_x, start_y), (goal_x, goal_y), grid_matrix

start_coord, goal_coord, grid_matrix = read_input_file("input2.txt")
print(start_coord)
print(goal_coord)

# Provides visualization of the grids prior to robot path exploration
flat_grid = [cell for row in grid_matrix for cell in row]
df = pd.DataFrame(flat_grid, columns=['x', 'y', 'cell_value'])

color_scale = alt.Scale(domain=[0, 1, 2, 5], range=['white', 'black', 'red', 'green'])

heatmap = alt.Chart(df).mark_rect().encode(
    x=alt.X('x:O', axis=alt.Axis(title='x', values=list(range(0, 51, 1)))),
    y=alt.Y('y:O', axis=alt.Axis(title='y', values=list(range(0, 31, 1))), sort='descending'),
    color=alt.Color('cell_value:N', scale=color_scale)
).properties(
    width=800,
    height=400
)

horizontal_grid = alt.Chart(pd.DataFrame({'y': [i + 0.5 for i in range(31)]})).mark_rule(color='gray', strokeWidth=0.5).encode(
    y=alt.Y('y:O', sort='descending') 
)

vertical_grid = alt.Chart(pd.DataFrame({'x': [i + 0.5 for i in range(51)]})).mark_rule(color='gray', strokeWidth=0.5).encode(
    x='x:O'
)

heatmap = heatmap + horizontal_grid + vertical_grid
heatmap

# Defines Node class
class Node:
    def __init__(self, coord, parent=None):
        self.coord = coord
        self.parent = parent
        self.g = 0 
        self.h = 0 
        self.f = 0 

# Defines a set of robot actions
actions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# Returns whether a move is legal
def is_legal_move(x, y, grid_dict):
    return grid_dict.get((x, y), 1) == 0 or grid_dict.get((x, y), 1) == 2 or grid_dict.get((x, y), 1) == 5

# Calculates euclidean distance for the heuristic function
def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Implements A* search algorithm
def a_star_search(start_coord, goal_coord, grid_dict):
    # Initialize the node count and visited count
    node_count = 0
    visited_count = 0

    # Defines a list of nodes that are yet to be explored
    frontier = []

    # Defines a set of visited nodes
    visited = set()

    # Defines start node and append it to the frontier
    start_node = Node(start_coord)
    start_node.h = euclidean_distance(start_coord, goal_coord)
    start_node.f = start_node.h + start_node.g
    frontier.append(start_node)
    
    while frontier:
        # Finds the node in the frontier with minimal cost
        current_node = min(frontier, key=lambda o: o.f)
        frontier.remove(current_node)
        visited.add(current_node.coord)
        visited_count += 1  # Increment visited count

        # Checks whether the goal node has been reached
        if current_node.coord == goal_coord:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = current_node.parent
            return path[::-1],node_count, visited_count

        # Explores child nodes by taking different actions
        for action in actions:
            new_coord = (current_node.coord[0] + action[0], current_node.coord[1] + action[1])
            # Avoids illegal moves or repeated states 
            if not is_legal_move(new_coord[0], new_coord[1], grid_dict) or new_coord in visited:
                continue
            new_node = Node(new_coord, current_node)
            node_count += 1
            new_node.g = current_node.g + (math.sqrt(2) if action in [(1, 1), (-1, 1), (-1, -1), (1, -1)] else 1)
            new_node.h = euclidean_distance(new_coord, goal_coord)
            new_node.f = new_node.g + new_node.h
            # Avoids adding new node if there is a node in the frontier with same coordinates and a lower or equal path cost
            if any(node.coord == new_node.coord and node.g <= new_node.g for node in frontier):
                continue
            frontier.append(new_node)
    # If no path is found, return None for the path but still return the counts
    return None, 


# Creates a grid dictionary for searching grid values
grid_dict = {(x, y): value for row in grid_matrix for x, y, value in row}

# Calls A* search to find the optimal path give start and goal coordinates
path, node_count, visited_count = a_star_search(start_coord, goal_coord, grid_dict)




if path:
    path_coords = [(node.coord[0], node.coord[1]) for node in path]
    write_output_file(path, start_coord, goal_coord, grid_dict, node_count, visited_count, "output.txt")
    
    # Visualization code here
    path_df = pd.DataFrame(path_coords, columns=['x', 'y'])
else:
    print("No path found")
    
    
'''
# Provides visualization of the grids with explored robot path
flat_grid = [cell for row in grid_matrix for cell in row]
df = pd.DataFrame(flat_grid, columns=['x', 'y', 'cell_value'])

color_scale = alt.Scale(domain=[0, 1, 2, 5], range=['white', 'black', 'red', 'blue'])

heatmap = alt.Chart(df).mark_rect().encode(
    x=alt.X('x:O', axis=alt.Axis(title='x', values=list(range(0, 51, 1)))),
    y=alt.Y('y:O', axis=alt.Axis(title='y', values=list(range(0, 31, 1))), sort='descending'),
    color=alt.Color('cell_value:N', scale=color_scale)
).properties(
    width=800,
    height=400
)

path_df = pd.DataFrame(path, columns=['x', 'y'])
path_map = alt.Chart(path_df).mark_point(color='green', filled=True, size=50).encode(
    x=alt.X('x:O', axis=alt.Axis(title='x', values=list(range(0, 51, 1)))),
    y=alt.Y('y:O', axis=alt.Axis(title='y', values=list(range(0, 31, 1))), sort='descending')
)

horizontal_grid = alt.Chart(pd.DataFrame({'y': [i + 0.5 for i in range(31)]})).mark_rule(color='gray', strokeWidth=0.5).encode(
    y=alt.Y('y:O', sort='descending') 
)

vertical_grid = alt.Chart(pd.DataFrame({'x': [i + 0.5 for i in range(51)]})).mark_rule(color='gray', strokeWidth=0.5).encode(
    x='x:O'
)

heatmap = heatmap + horizontal_grid + vertical_grid + path_map
heatmap

'''





