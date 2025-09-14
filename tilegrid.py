import heapq
import time

class PNode:
    # Represents one instance of the class
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost  
        self.heuristic = 0          
        self.total_cost = 0         

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

def find_blank(state):
    for r, row in enumerate(state):
        for c, val in enumerate(row):
            if val == 0:
                return r, c
    return None

def get_actions(state):
    actions = []
    r, c = find_blank(state)
    # Safeguard to keep pieces on the board
    if r > 0: actions.append('Up')
    if r < 2: actions.append('Down')
    if c > 0: actions.append('Left')
    if c < 2: actions.append('Right')
    return actions

def apply_action(state, action):
    """Applies an action to a state to get a new state."""
    r, c = find_blank(state)
    new_state = [list(row) for row in state] # Convert to mutable list of lists

    if action == 'Up':
        new_state[r][c], new_state[r-1][c] = new_state[r-1][c], new_state[r][c]
    elif action == 'Down':
        new_state[r][c], new_state[r+1][c] = new_state[r+1][c], new_state[r][c]
    elif action == 'Left':
        new_state[r][c], new_state[r][c-1] = new_state[r][c-1], new_state[r][c]
    elif action == 'Right':
        new_state[r][c], new_state[r][c+1] = new_state[r][c+1], new_state[r][c]

    return tuple(tuple(row) for row in new_state)

## Heuristic Functions
def h0_misplaced_tiles(state, goal_state):
    """Counts how many tiles are out of their goal position."""
    return sum(1 for r in range(3) for c in range(3) if state[r][c] != 0 and state[r][c] != goal_state[r][c])

def h1_row_col_placement(state, goal_state):
    """Counts how many tiles are in the wrong row or wrong column."""
    score = 0
    goal_positions = {tile: (r, c) for r, row in enumerate(goal_state) for c, tile in enumerate(row)}

    for r, row in enumerate(state):
        for c, tile in enumerate(row):
            if tile != 0:
                goal_r, goal_c = goal_positions[tile]
                if r != goal_r:
                    score += 1
                if c != goal_c:
                    score += 1
    return score

def h2_manhattan_distance(state, goal_state):
    """Calculates the sum of Manhattan distances for all tiles."""
    distance = 0
    goal_positions = {tile: (r, c) for r, row in enumerate(goal_state) for c, tile in enumerate(row)}

    for r, row in enumerate(state):
        for c, tile in enumerate(row):
            if tile != 0:
                goal_r, goal_c = goal_positions[tile]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def solve_puzzle(initial_state, goal_state, heuristic):
    start_node = PNode(state=initial_state, path_cost=0)
    start_node.heuristic = heuristic(initial_state, goal_state)
    start_node.total_cost = start_node.path_cost + start_node.heuristic

    frontier = [start_node]
    explored = set()
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        
        current_node = heapq.heappop(frontier)

        if current_node.state == goal_state:
            path = reconstruct_path(current_node)
            score = current_node.path_cost # The score is the number of moves
            return path, score, max_frontier_size

        explored.add(current_node.state)

        for action in get_actions(current_node.state):
            child_state = apply_action(current_node.state, action)
            
            if child_state in explored:
                continue

            child_node = PNode(
                state=child_state,
                parent=current_node,
                action=action,
                path_cost=current_node.path_cost + 1 
            )
            child_node.heuristic = heuristic(child_state, goal_state)
            child_node.total_cost = child_node.path_cost + child_node.heuristic

            if not any(node.state == child_state and node.total_cost <= child_node.total_cost for node in frontier):
                heapq.heappush(frontier, child_node)
    
    return None, None, max_frontier_size

def reconstruct_path(node):
    """Traces back from the goal node to the start node to find the path."""
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    return list(reversed(path))

if __name__ == "__main__":
    solvable_puzzles = [
    # Mild Difficulty
    ((1, 2, 3), (4, 5, 0), (7, 8, 6)),
    ((1, 2, 3), (4, 5, 6), (0, 7, 8)),
    ((1, 2, 3), (4, 5, 6), (7, 0, 8)),
    ((1, 2, 3), (0, 5, 6), (4, 7, 8)),
    ((1, 5, 2), (4, 0, 3), (7, 8, 6)),
    ((1, 2, 3), (4, 8, 5), (7, 0, 6)),
    ((4, 1, 3), (7, 2, 5), (0, 8, 6)),
    ((1, 2, 0), (4, 5, 3), (7, 8, 6)),
    ((4, 1, 2), (7, 5, 3), (0, 8, 6)),
    ((1, 3, 0), (4, 2, 5), (7, 8, 6)),
    ((4, 1, 2), (5, 8, 3), (7, 6, 0)),
    ((5, 1, 3), (4, 0, 2), (7, 8, 6)),
    ((2, 4, 3), (1, 5, 6), (7, 8, 0)),
    ((1, 8, 2), (0, 4, 3), (7, 6, 5)),
    ((4, 1, 2), (7, 5, 3), (8, 6, 0)),
    ((7, 1, 2), (4, 8, 5), (0, 6, 3)),
    ((8, 6, 7), (2, 5, 4), (3, 0, 1)),
    ((8, 2, 7), (4, 0, 6), (5, 1, 3)),
    ((4, 1, 2), (0, 8, 3), (5, 7, 6)),
    ((3, 6, 5), (4, 1, 2), (0, 8, 7)),
    ((7, 4, 0), (1, 8, 5), (3, 2, 6)),
    ((0, 5, 6), (4, 1, 3), (2, 7, 8)),
    ((1, 6, 3), (4, 8, 2), (7, 0, 5)),
    ((2, 0, 3), (1, 5, 6), (4, 7, 8)),
    ((7, 1, 3), (0, 5, 2), (4, 8, 6)),
    ((5, 2, 0), (1, 8, 3), (4, 7, 6)),
    ((1, 8, 3), (4, 6, 5), (7, 0, 2)),
    ((2, 1, 3), (4, 6, 5), (7, 0, 8)),
    ((2, 1, 3), (0, 6, 5), (4, 7, 8)),
    ((4, 2, 3), (1, 6, 5), (7, 0, 8)),
]
    goal_state = (
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 0)
    )

    # Dictionary to hold the heuristic functions for easy testing
    heuristics_to_test = {
        "Misplaced Tiles": h0_misplaced_tiles,
        "Row/Column Placement": h1_row_col_placement,
        "Manhattan Distance": h2_manhattan_distance
    }
    

for i, initial_state in enumerate(solvable_puzzles):
        print(f"\n----- TESTING PUZZLE #{i+1} -----")
        print(initial_state)
        for name, heuristic in heuristics_to_test.items():
            print(f"  Running with heuristic: '{name}'...")
            start_time = time.time()
            
            solution_path, score, max_frontier = solve_puzzle(initial_state, goal_state, heuristic)
            
            end_time = time.time()

            if solution_path:
                print(f"    Solution Found! Score: {score}, Max Frontier: {max_frontier}, Time: {end_time - start_time:.4f}s")
            else:
                print("    No solution found.")