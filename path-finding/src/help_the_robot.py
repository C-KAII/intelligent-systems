# Intelligent Systems - "Help the Robot" Pathfinding Project
# Designed and developed by Kobi Chambers - Griffith University

# Import necessary relevent packages and modules
try:
    import os
    import sys
    import time
    import random
    import tkinter as tk
    from tkinter import filedialog
    from collections import deque
    from queue import PriorityQueue
    from heapq import heappush, heappop
except ImportError as e:
    print(f"Error importing module: {e}")
    print(f"Please ensure that module is installed...")
    sys.exit(1)

try:
    no_tqdm_flag = False
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing module: {e}")
    no_tqdm_flag = True
    print(f"\nInstall {e} library to display progress bars, or run in terminal if already installed & still getting this exception message...")

# Define movements dictionary with their associated costs and move symbols for each
MOVES_DICTIONARY = {
    # Diagonal movement has cost 1
    (1, 1): (1, 'DR'),  # down-right movement
    (-1, 1): (1, 'UR'),  # up-right movement
    (1, -1): (1, 'DL'),  # down-left movement
    (-1, -1): (1, 'UR'),  # up-left movement
    # Straight movement has cost 2
    (0, 1): (2, 'R'),   # right movement
    (0, -1): (2, 'L'),  # left movement
    (1, 0): (2, 'D'),   # down movement
    (-1, 0): (2, 'U')   # up movement
}

# Create a tk window and hide it
root = tk.Tk()
root.withdraw()

# Get the absolute path of the folder containing the program file
program_folder = os.path.dirname(os.path.abspath(__file__))

# Get filepath from user, initial directory is within the same folder as program
input_file_path = filedialog.askopenfilename(
    initialdir=program_folder, title="Please select the input txt file", filetypes=(("Text files", "*.txt"),))

# Check that user selected a file
if input_file_path:
    # Split base name to find input filename without path or extension
    input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    # Print to show the user which input was selected, and the output file created
    print(f"\nFile selected: {input_file_name}\n")
else:
    sys.exit("User closed file dialogue... exiting program.\n")

####################################
### SETUP AND CHECKING FUNCTIONS ###
####################################


def timer(function_name):
    """
    Decorator function to measure execution time of a function
    The function results are returned, along with the elapsed time to process
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function to measure execution time of a function
        The function results are returned, along with the elapsed time to process
        """
        start_time = time.perf_counter()
        result = function_name(*args, **kwargs)
        end_time = time.perf_counter()

        # Convert to microseconds
        elapsed_time = (end_time - start_time) * 1_000_000

        return result, elapsed_time

    return wrapper


def versions_window():
    """
    Function to allow user to select HCS version
    """
    root = tk.Tk()
    root.title("Select a version")
    root.geometry("300x200")

    # Initialise selected version variable
    selected_version = tk.StringVar()

    def set_selected_option(version):
        """
        Function to update the selected_version variable
        """
        # Close window
        root.destroy()
        # Set selected version
        selected_version.set(version)

    # Create the label and buttons for the version selections
    label = tk.Label(root, text="Please select a HCS algorithm version:")
    button1 = tk.Button(root, text="Greedy HCS",
                        command=lambda: set_selected_option("Greedy"))
    button2 = tk.Button(root, text="Random Restart HCS",
                        command=lambda: set_selected_option("Random_Restart"))
    button3 = tk.Button(root, text="Randomised HCS",
                        command=lambda: set_selected_option("Randomised"))

    # Add widgets
    label.pack(pady=10)
    button1.pack(pady=5)
    button2.pack(pady=5)
    button3.pack(pady=5)

    # Wait for the user to select a version
    root.wait_window()

    # Return selected version
    return selected_version.get()


def handle_error(error, algorithm_number=None, board_size_N=None, row_length=None, board_height=None):
    """
    Function to handle input errors, mainly for code cleanliness
    Parsed values overwrite the Nonetypes
    Prompts user for new input and exits program
    """
    match error:
        case 'algorithm_error':
            sys.exit(f"Error:\tInput text file: line 1... '{algorithm_number}' is an incorrect value for selecting an algorithm... \n"
                     "\t\t***Update input file at line 1 to a value between 1 and 5, and try again.***")

        case 'len_row_error':
            sys.exit(f"Error:\tInput text file: line 2 specifies board size, N = {board_size_N}, "
                     f"which does not match the # of columns in input board where N = {row_length}... \n"
                     f"\t\t***Update input file at line 2 or rest of file, and try again.***")

        case 'num_rows_error':
            sys.exit(f"Error:\tInput text file: line 2 specifies board size, N = {board_size_N}, "
                     f"which does not match the # of rows in input board where N = {board_height}... \n"
                     "\t\t***Update input file at line 2 or the number of rows, and try again.***")

        case 'input_format_error':
            sys.exit("Error:\tInput text file: input values at line 1 or 2 are not valid integers...\n"
                     "\t\t***Update input file at line 1 or 2, and try again.***")

        case 'start_error':
            sys.exit("Error:\tInput text file: no start point was found...\n"
                     "\t\t***Update input file board state, and try again.***")

        case 'goal_error':
            sys.exit("Error:\tInput text file: no goal point was found...\n"
                     "\t\t***Update input file board state, and try again.***")

        case _:
            sys.exit(f"Error:\t{error}, exiting program...\n")


def check_algorithm(algorithm_number):
    """
    Check which algorithm has been selected
    A string that better describes the algorithm chosen is returned
    """
    match algorithm_number:
        case 1:
            algorithm_selected = 'BFS'

        case 2:
            algorithm_selected = 'UCS'

        case 3:
            algorithm_selected = 'IDS'

        case 4:
            algorithm_selected = 'ASTAR'

        case 5:
            algorithm_selected = 'HCS'

        case _:
            handle_error('algorithm_error', algorithm_number=algorithm_number)

    return algorithm_selected


def process_input_file():
    """
    Function to process the lines within an input text file
    If no errors found, returns values defined by the input file lines as follows:
    algorithm_selected - First line
    board_size_N - Second line
    board - Rest of file
    start_point - 'S' found on the board
    goal_point - 'G' found on the board
    """
    start_point = None
    goal_point = None
    # 'with open' reads input file in as an iterable object, and automatically closes the file when done reading
    with open(input_file_path) as f:
        try:
            # Read first line and set the algorithm number input
            algorithm_number = int(f.readline().strip())
            # Read second line and set the board size, N
            board_size_N = int(f.readline().strip())
        except:
            f.close()
            handle_error('input_format_error')

        # Define empty board
        board = []

        # Create board from rest of file
        for lines in f.readlines():
            # Define empty row
            row = []
            # Strip lines for a clean string
            line = lines.strip()
            # Iterate through each character in line string
            for cell in line:
                if cell == 'S':
                    # Set start point index
                    start_point = (len(board), len(row))
                elif cell == 'G':
                    # Set goal point index
                    goal_point = (len(board), len(row))
                # Append cells to row to create 1D list
                row.append(cell)

            # Check for row length error, close input.txt and throw error
            if len(row) != board_size_N:
                f.close()
                handle_error('len_row_error',
                             board_size_N=board_size_N, row_length=len(row))
            else:
                # Append rows to board to create 2D array
                board.append(row)

        # Check if start and goal points are defined, if not, handle error
        if start_point is None:
            f.close()
            handle_error('start_error')
        if goal_point is None:
            f.close()
            handle_error('goal_error')
    # Input text file is now closed

    # Identify the algorithm selected and error check the selection within function
    algorithm_selected = check_algorithm(algorithm_number)
    # Check for column length error
    if len(board) != board_size_N:
        handle_error('num_rows_error', board_size_N=board_size_N,
                     board_height=len(board))

    # If all ok, return values
    return algorithm_selected, board_size_N, board, start_point, goal_point

####################################
### UTILITY AND HELPER FUNCTIONS ###
####################################


def diag_dist_heuristic(current_position, goal_point):
    """
    Diagonal Distance Heuristic Function
    """
    # Get the absolute distance between the goal x, y coords and current position x, y cords respectively.
    dx = abs(current_position[0] - goal_point[0])
    dy = abs(current_position[1] - goal_point[1])

    # Return heuristic value as below since diagonal moves cost 1 and cardinal moves cost 2
    return min(dx, dy) * 1 + max(dx, dy) * 2 - min(dx, dy) * 2


def get_valid_neighbours(current_position, discovered, board_size_N, board):
    """
    Generate a list of the valid neighbouring cells from the current position
    """
    valid_neighbours = []
    # Loop through possible movements
    for move in MOVES_DICTIONARY:
        # Set the check flag for loop
        check_legal = False
        # Set the next cell position
        next_position = (
            current_position[0] + move[0], current_position[1] + move[1])
        # Check if legal move, and not discovered
        if all(0 <= next_position[i] < board_size_N for i in range(2)) and next_position not in discovered and board[next_position[0]][next_position[1]] != 'X':
            # Using match move to identify the additional illegal moves if moving diagonally
            match move:
                case (1, 1):
                    if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                        check_legal = True

                case (-1, 1):
                    if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                        check_legal = True

                case (1, -1):
                    if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                        check_legal = True

                case(-1, -1):
                    if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                        check_legal = True
                # Default case is for all other non diagonal moves, as these are already identified as legal
                case _:
                    check_legal = True

        # Finally, check if the move is legal
        if check_legal is True:
            # If it is, concatenate next position to valid_neighbours list
            valid_neighbours.append(next_position)

    return valid_neighbours


def get_random_start(solution_length_M, board_size_N, board, start_point):
    """
    Generate a random starting sequence and path
    """
    # Initialise empty random sequence and initialise random path with start point
    random_sequence = []
    random_path = [start_point]
    # Generate random sequence and path
    for _ in range(solution_length_M):
        # Set current position
        current_pos = random_path[-1]
        # List valid valid_neighbours
        valid_neighbours = get_valid_neighbours(
            current_pos, random_path, board_size_N, board)
        # Check if there are still moves to make
        if valid_neighbours:
            # Use Python random module to select a random neighbour from the valid list
            random_neighbour = random.choice(valid_neighbours)
            # Convert the neighbour position into a move
            random_move = (
                random_neighbour[0] - current_pos[0], random_neighbour[1] - current_pos[1])
            # Calculate the next position
            next_pos = (current_pos[0] + random_move[0],
                        current_pos[1] + random_move[1])
            # Add the next position to the random sequence
            random_sequence.append(random_move)
            random_path.append(next_pos)
        else:
            # Break if no valid moves
            break

    return random_sequence, random_path


def get_fitness(current_position, board_size_N, board, goal_point, sequence):
    """
    Calculate the fitness of a solution
    """
    # Initialise fitness and discovered set
    fitness = 0
    discovered = set(start_point)
    heuristic_value = 0

    # Loop through moves in the random sequence
    for move in sequence:
        # Calculate the next position
        next_position = (
            current_position[0] + move[0], current_position[1] + move[1])

        # Check if next position is a valid neighbour
        if next_position in get_valid_neighbours(current_position, discovered, board_size_N, board):
            # Set new position and increment the fitness value
            current_position = next_position
            fitness += 1
            discovered.add(current_position)
            # Add to the heuristic value
            heuristic_value += diag_dist_heuristic(
                current_position, goal_point)
        else:
            break

    # Check if the sequence reached the goal
    if current_position == goal_point:
        fitness += 100

    # Return the calculated fitness value
    return fitness, heuristic_value, discovered

##################################
### SEARCH ALGORITHM FUNCTIONS ###
##################################


def execute_bfs(algorithm_selected, board_size_N, board, start_point, goal_point):
    """
    Function to execute the Breadth First Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns explored states
    If no path, returns explored states
    """
    # Initialise a queue to track next visits
    queue = deque()

    # Initialise set to track discovered cells and count
    discovered = set()
    explored_states = 0

    # Set start cell
    queue.append(start_point)
    discovered.add(start_point)

    # Define path dict to track path to reach each cell
    paths = {start_point: None}

    # Loop until all cells discovered or end cell discovered
    while queue:
        # Pop cell from queue
        current_position = queue.popleft()
        # Check if goal reached
        if current_position == goal_point:
            # Reconstruct the path
            return path_found(algorithm_selected, paths, start_point, goal_point), explored_states, discovered

        # Loop through possible movements
        for move in MOVES_DICTIONARY:
            # Set the check flag for loop
            check_legal = False
            # Set the next cell position
            next_position = (
                current_position[0] + move[0], current_position[1] + move[1])
            # Check if legal move, and not discovered
            if all(0 <= next_position[i] < board_size_N for i in range(2)) and next_position not in discovered and board[next_position[0]][next_position[1]] != 'X':
                # Using match move to identify the additional illegal moves if moving diagonally
                match move:
                    case (1, 1):
                        if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                            check_legal = True

                    case (-1, 1):
                        if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                            check_legal = True

                    case (1, -1):
                        if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                            check_legal = True

                    case(-1, -1):
                        if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                            check_legal = True

                    # Default case is for all other non diagonal moves, as these are already identified as legal
                    case _:
                        check_legal = True

            # Finally, check if the move is legal
            if check_legal is True:
                # If it is, append next pos to queue and add to discovered, then add the path
                queue.append(next_position)
                discovered.add(next_position)
                paths[next_position] = current_position
                explored_states += 1

            # Reset the check for next loop
            check_legal = False

    # Return None and explored states if path not found
    return None, explored_states, discovered


def execute_ucs(algorithm_selected, board_size_N, board, start_point, goal_point):
    """
    Function to execute the Uniform Cost Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns explored states
    If no path, returns explored states
    """
    # Initialise a priority queue to track next visits
    priority_q = PriorityQueue()
    # Initialise dict to track paths and costs, initialise a discovered (purely for visualisation)
    paths = {}
    discovered = set()
    # Simple counter for states explored, not including start point
    explored_states = 0
    # Set start cell
    priority_q.put((0, start_point))
    paths[start_point] = {'cost': 0, 'parent': None}

    # Loop until all cells discovered or end cell discovered
    while not priority_q.empty():
        # get cell from queue and add to discovered
        current_cost, current_position = priority_q.get()
        discovered.add(current_position)

        # Check if goal reached
        if current_position == goal_point:
            # Reconstruct the path
            return path_found(algorithm_selected, paths, start_point, goal_point), explored_states, discovered

        # Loop through possible movements
        for move in MOVES_DICTIONARY:
            # Set the check flag for loop
            check_legal = False
            # Set the next cell position
            next_position = (
                current_position[0] + move[0], current_position[1] + move[1])
            # Check if legal move, and not discovered
            if all(0 <= next_position[i] < board_size_N for i in range(2)) and next_position not in paths and board[next_position[0]][next_position[1]] != 'X':
                # Using match move to identify the additional illegal moves if moving diagonally
                match move:
                    case (1, 1):
                        if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                            check_legal = True

                    case (-1, 1):
                        if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                            check_legal = True

                    case (1, -1):
                        if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                            check_legal = True

                    case(-1, -1):
                        if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                            check_legal = True

                    # Default case is for all other non diagonal moves, as these are already identified as legal
                    case _:
                        check_legal = True

            # Finally, check if the move is legal
            if check_legal:
                # If it is, calculate new cost from moves dict, then add the path if cost is lower
                new_cost = current_cost + MOVES_DICTIONARY[move][0]

                # Check if new path is lower cost
                if next_position not in paths or new_cost < paths[next_position]['cost']:
                    paths[next_position] = {
                        'cost': new_cost, 'parent': current_position}
                    # Add to pq with new cost
                    priority_q.put((new_cost, next_position))
                    explored_states += 1

    # Return None and explored states if path not found
    return None, explored_states, discovered


def execute_ids(algorithm_selected, board_size_N, board, start_point, goal_point):
    """
    Function to execute the Iterative Deepening Search algorithm on the input board
    Repeated calls the limited_depth_search helper function
    If no path found, returns explored states
    """

    def limited_depth_search(depth_limit, board_size_N, board, current_position, goal_point, path):
        """
        Recursive function to execute the limited depth search (DFS) algorithm on the input board at the specified depth limit
        Utilises recursion to find a solution path, if it exists at current depth
        If we reach the depth limit or no path found, returns explored states
        """
        # Check if goal reached
        if current_position == goal_point:
            path[current_position] = []
            return path

        # Check if depth limit reached
        elif depth_limit == 0:
            return None

        # Else continue through iterations
        else:
            # Add current position to discovered set
            discovered.add(current_position)

            # Loop through possible movements
            for move in MOVES_DICTIONARY:
                # Set the check flag for loop
                check_legal = False
                # Set the next cell position
                next_position = (
                    current_position[0] + move[0], current_position[1] + move[1])
                # Check if legal move, and not discovered
                if all(0 <= next_position[i] < board_size_N for i in range(2)) and next_position not in discovered and board[next_position[0]][next_position[1]] != 'X':
                    # Using match move to identify the additional illegal moves if moving diagonally
                    match move:
                        case (1, 1):
                            if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                                check_legal = True

                        case (-1, 1):
                            if board[current_position[0]][current_position[1] + 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                                check_legal = True

                        case (1, -1):
                            if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] + 1][current_position[1]] != 'X':
                                check_legal = True

                        case(-1, -1):
                            if board[current_position[0]][current_position[1] - 1] != 'X' and board[current_position[0] - 1][current_position[1]] != 'X':
                                check_legal = True

                        # Default case is for all other non diagonal moves, as these are already identified as legal
                        case _:
                            check_legal = True

                    # Finally, check if the move is legal
                    if check_legal is True:
                        # Recursively call for next positions
                        new_path = limited_depth_search(
                            depth_limit - 1, board_size_N, board, next_position, goal_point, path)

                        # Check if paths is not None
                        if new_path is not None and new_path[next_position] is not None:
                            path[current_position] = [
                                current_position] + new_path[next_position]
                            return path

                        discovered.add(next_position)

            # Return None, and length of explored states if no path
            return None

    total_discovered = set()
    # Initialise a progress bar
    if no_tqdm_flag is False:
        print("Depth level is only an estimate. This could take much longer as progress slows exponentially...")
        pbar = tqdm(total=board_size_N**2, desc="Depth", unit=" depth level")

    # Setting the depth limit to N^2 to ensure every possible path is explored from start point
    for depth_limit in range(1, board_size_N**2):
        discovered = set()

        # Update progress bar
        if no_tqdm_flag is False:
            pbar.update(1)
        else:
            print(f"Current depth being searched: {depth_limit}")

        # Call limited depth search (DFS) function for each depth limit value
        paths = limited_depth_search(
            depth_limit, board_size_N, board, start_point, goal_point, {})

        # Union total with discovered sets
        total_discovered |= discovered

        if paths is not None and goal_point in paths:
            # Close progress bar
            if no_tqdm_flag is False:
                pbar.close()
            # Reconstruct the path
            return path_found(algorithm_selected, paths[start_point], start_point, goal_point), len(total_discovered), total_discovered
    # Close progress bar
    if no_tqdm_flag is False:
        pbar.close()
    return None, len(total_discovered), total_discovered


def execute_astar(algorithm_selected, board_size_N, board, start_point, goal_point):
    """
    Function to execute the A Star Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns explored states
    If no path, returns explored states
    """
    # Initialise explored states counter, discovered set and a priority queue with f value, start point and path
    explored_states = 0
    discovered = set([start_point])
    queue = [(0 + diag_dist_heuristic(start_point, goal_point), start_point, [])]

    while queue:
        # Pop lowest f value and corresponding current position and path
        f_value, current_position, path = heappop(queue)

        # Check if we've reached the goal
        if current_position == goal_point:
            return path_found(algorithm_selected, path, start_point, goal_point), explored_states, discovered

        # Explore valid neighbours
        for neighbour in get_valid_neighbours(current_position, discovered, board_size_N, board):
            if neighbour:
                discovered.add(neighbour)

                # Utilise sequence and cost function to find g_value which is the path so far cost, not interested in the sequence
                _, g_value = sequence_and_cost(path)
                # Calculate the new f value (cost so far + heuristic value)
                f_value = g_value + diag_dist_heuristic(neighbour, goal_point)

                # Add newly discovered position and corresponding values to the queue
                heappush(queue, (f_value, neighbour, path + [neighbour]))
                explored_states += 1

    # No path found, return None and explored states
    return None, explored_states, discovered


def execute_hcs_greedy(board_size_N, board, start_point, goal_point, random_sequence, bfs_solution):
    """
    Function to execute the Greedy Hill Climb Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns steps to end
    If stuck in local max/min, returns steps to end
    """
    # Initialise current position with start point
    current_position = start_point
    steps_to_end = 0
    discovered = set()
    # Define MAX_STEPS constant
    MAX_STEPS = 20

    # This ensures we have a sequence to start with and know the length of an actual solution sequence
    # Find the sequence for the bfs solution
    bfs_sequence = []
    solution_length_M = len(bfs_solution) - 1

    for x in range(solution_length_M):
        # Find what the move was for this step
        solution_move = tuple(
            [bfs_solution[x + 1][y] - bfs_solution[x][y] for y in range(2)])
        # Lookup dict. move value at index 1, then append to list
        bfs_sequence.append(solution_move)

    # Initialise best_sequence with the random_sequence parsed
    best_sequence = random_sequence

    # Calculate initial fitness of random sequence
    current_fitness, current_heuristic, temp_discovered = get_fitness(
        current_position, board_size_N, board, goal_point, best_sequence)
    discovered |= temp_discovered

    # Loop possible solutions until no better solutions
    while steps_to_end < MAX_STEPS:
        # Iterate steps to end
        steps_to_end += 1

        # Create list of neighbours from making a single move in all directions
        # While this seems like it may not be a greedy approach, as we find all neighbours, we only take the best later on
        neighbourhood = [best_sequence[:i] + [next_move] + best_sequence[i + 1:]
                         for i in range(len(bfs_sequence)) for next_move in MOVES_DICTIONARY]

        # Find fitness & heuristics of each neighbour
        all_fitnesses = []
        all_heuristics = []
        for neighbour in neighbourhood:
            x, y, temp_discovered = get_fitness(
                current_position, board_size_N, board, goal_point, neighbour)
            all_fitnesses.append(x)
            all_heuristics.append(y)

        # Select best solution from max fitness
        best_fitness = max(all_fitnesses)

        # Check if best > current
        if best_fitness > current_fitness:
            index = all_fitnesses.index(best_fitness)
            best_sequence = neighbourhood[index]
            current_fitness = best_fitness
            discovered |= temp_discovered

        # Check if best == current
        elif best_fitness == current_fitness:
            # As there are multiple values of best fitness that equal, we should compare heuristic values to attempt to get closer to the goal
            # Set indices of the best values
            best_heuristics = []
            best_fit_indices = [i for i, x in enumerate(
                all_fitnesses) if x == best_fitness]
            # Find best heuristics out of the best fitness indices
            best_heuristics = [all_heuristics[i] for i in best_fit_indices]
            # Find lowest heuristic value in the list
            best_heuristic = min(best_heuristics)

            # Check if best heuristic < current
            if best_heuristic < current_heuristic:
                # Update the best sequence and new best heuristic
                index = best_heuristics.index(best_heuristic)
                best_sequence = neighbourhood[index]
                current_heuristic = best_heuristic
                discovered |= temp_discovered
            else:
                break

        # Else, we are on the best path we can be
        else:
            break

    # Check if the final position from the best sequence has reached the goal
    for move in best_sequence:
        current_position = (
            current_position[0] + move[0], current_position[1] + move[1])
    if current_position == goal_point:
        print("Path found\n")
        return path_found_hcs(best_sequence, start_point, goal_point), steps_to_end, discovered
    # If not, the algorithm became stuck in a local min or max
    else:
        print("Local min/max has been reached\n")
        return path_found_hcs(best_sequence, start_point, goal_point), steps_to_end, discovered


def execute_hcs_random_restart(board_size_N, board, start_point, goal_point, bfs_solution):
    """
    Function to execute the Random Restart Hill Climb Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns steps to end
    If stuck in local max/min, returns steps to end
    """
    # Define MAX_STEPS and MAX_RESTARTS constants
    MAX_RESTARTS = 10
    MAX_STEPS = 20

    # Initialise a progress bar
    if no_tqdm_flag is False:
        pbar = tqdm(total=MAX_RESTARTS*MAX_STEPS, desc="Steps", unit=" loops")

    # Initialise variables to track the best solutions across each step and each random restart
    current_fitness = float('-inf')
    current_heuristic = float('inf')
    best_fitness = float('-inf')
    best_heuristic = float('inf')
    best_solution = None
    discovered = set()

    # This ensures we have a sequence to start with and know the length of an actual solution sequence
    # Find the sequence for the bfs solution
    bfs_sequence = []
    solution_length_M = len(bfs_solution) - 1

    for x in range(solution_length_M):
        # Find what the move was for this step
        solution_move = tuple(
            [bfs_solution[x + 1][y] - bfs_solution[x][y] for y in range(2)])
        # Lookup dict. move value at index 1, then append to list
        bfs_sequence.append(solution_move)

    # Loop to MAX_RESTARTS
    for _ in range(MAX_RESTARTS):
        # Initialise current position with start point, and steps to end
        current_position = start_point
        steps_to_end = 0

        # Initialise random_path and loop to ensure we find a sequence of moves that can reach the goal point
        random_path = []
        while len(random_path) < len(bfs_solution):
            best_sequence, random_path = get_random_start(
                solution_length_M, board_size_N, board, start_point)

        # Calculate initial fitness of random sequence
        current_fitness, current_heuristic, temp_discovered = get_fitness(
            current_position, board_size_N, board, goal_point, best_sequence)
        discovered |= temp_discovered

        # Loop possible solutions until no better solutions
        while steps_to_end < MAX_STEPS:
            # Iterate steps to end
            steps_to_end += 1

            # Update progress bar
            if no_tqdm_flag is False:
                pbar.update(1)

            # Create list of neighbours from making a single move in all directions
            neighbourhood = [best_sequence[:i] + [next_move] + best_sequence[i + 1:]
                             for i in range(len(bfs_sequence)) for next_move in MOVES_DICTIONARY]

            # Find fitness & heuristics of each neighbour
            all_fitnesses = []
            all_heuristics = []
            # Looped manually instead of list comprehension to easier unpack values
            for neighbour in neighbourhood:
                x, y, temp_discovered = get_fitness(
                    current_position, board_size_N, board, goal_point, neighbour)
                all_fitnesses.append(x)
                all_heuristics.append(y)

            # Select best current fitness from max fitness
            temp_best_fitness = max(all_fitnesses)

            # Check if best > current
            if temp_best_fitness > current_fitness:
                index = all_fitnesses.index(temp_best_fitness)
                best_sequence = neighbourhood[index]
                current_fitness = temp_best_fitness
                discovered |= temp_discovered

            # Check if best fit == current fit
            elif temp_best_fitness == current_fitness:
                # As there are multiple values of best fitness that equal, we should now compare heuristic values to attempt to get closer to the goal
                # Set indices of the best values
                best_heuristics = []
                best_fit_indices = [i for i, x in enumerate(
                    all_fitnesses) if x == temp_best_fitness]
                # Find best heuristics out of the best fitness indices
                best_heuristics = [all_heuristics[i] for i in best_fit_indices]
                # Find lowest heuristic value in the list
                temp_best_heuristic = min(best_heuristics)

                # Check if best heuristic < current
                if temp_best_heuristic < current_heuristic:
                    # Update the best sequence and new best heuristic
                    index = best_heuristics.index(temp_best_heuristic)
                    best_sequence = neighbourhood[index]
                    current_heuristic = temp_best_heuristic
                    discovered |= temp_discovered

            # Check if we've made improvements since last restart and random sequence
            if current_fitness > best_fitness or (current_fitness == best_fitness and current_heuristic < best_heuristic):
                # Set new best values
                best_fitness = current_fitness
                best_heuristic = current_heuristic
                best_solution = best_sequence[:]
                best_steps_to_end = steps_to_end

        # Continue to next restart loop
        else:
            continue

    # Close progress bar
    if no_tqdm_flag is False:
        pbar.close()

    # Check if we found a best solution
    if best_solution:
        # Check if the final position from the best sequence has reached the goal
        for move in best_sequence:
            current_position = (
                current_position[0] + move[0], current_position[1] + move[1])

        if current_position == goal_point:
            print("\nPath found\n")
            return path_found_hcs(best_solution, start_point, goal_point), best_steps_to_end, discovered

        else:
            print("\nLocal min/max has been reached\n")
            return path_found_hcs(best_solution, start_point, goal_point), best_steps_to_end, discovered

    else:
        print("\nLocal min/max has been reached\n")
        return path_found_hcs(best_sequence, start_point, goal_point), steps_to_end, discovered


def execute_hcs_randomised(board_size_N, board, start_point, goal_point, random_sequence, bfs_solution):
    """
    Function to execute the Randomised Hill Climb Search algorithm on the input board
    If path found, executes 'path_found' algorithm, and returns steps to end
    If stuck in local max/min, returns steps to end
    """
    # Initialise current position with start point
    current_position = start_point
    steps_to_end = 0
    discovered = set()
    # Define MAX_STEPS constant
    MAX_STEPS = 20

    # This ensures we have a sequence to start with and know the length of an actual solution sequence
    # Find the sequence for the bfs solution
    bfs_sequence = []
    solution_length_M = len(bfs_solution) - 1

    for x in range(solution_length_M):
        # Find what the move was for this step
        solution_move = tuple(
            [bfs_solution[x + 1][y] - bfs_solution[x][y] for y in range(2)])
        # Lookup dict. move value at index 1, then append to list
        bfs_sequence.append(solution_move)

    # Initialise best_sequence with the random_sequence parsed
    best_sequence = random_sequence

    # Calculate initial fitness of random sequence
    current_fitness, current_heuristic, temp_discovered = get_fitness(
        current_position, board_size_N, board, goal_point, best_sequence)
    best_fitness = current_fitness
    best_heuristic = current_heuristic
    discovered |= temp_discovered

    # Loop possible solutions until no better solutions
    while steps_to_end < MAX_STEPS:
        # Iterate steps to end
        steps_to_end += 1

        # Create list of neighbours from making a single move in all directions
        neighbourhood = [best_sequence[:i] + [next_move] + best_sequence[i + 1:]
                         for i in range(len(bfs_sequence)) for next_move in MOVES_DICTIONARY]

        # Randomly shuffle all neighbour sequences
        random.shuffle(neighbourhood)

        # Find fitness & heuristics of neighbours
        for neighbour in neighbourhood:
            x, y, temp_discovered = get_fitness(
                current_position, board_size_N, board, goal_point, neighbour)
            if x > best_fitness or (x == best_fitness and y < best_heuristic):
                best_sequence = list(neighbour)
                best_fitness = x
                best_heuristic = y
                # This discovered set should look slightly different on visuals
                discovered |= temp_discovered
                break

    # Check if the final position from the best sequence has reached the goal
    for move in best_sequence:
        current_position = (
            current_position[0] + move[0], current_position[1] + move[1])
    if current_position == goal_point:
        print("Path found\n")
        return path_found_hcs(best_sequence, start_point, goal_point), steps_to_end, discovered
    # If not, the algorithm became stuck in a local min or max
    else:
        print("Local min/max has been reached\n")
        return path_found_hcs(best_sequence, start_point, goal_point), steps_to_end, discovered

###################################
### EXECUTION UTILITY FUNCTIONS ###
###################################


def path_found(algorithm_selected, paths, start_point, goal_point):
    """
    Function to construct the path to reach the goal point
    Returns the solution path that the algorithm found
    """
    # Initialise the current position, i.e. goal point as a path was found
    current_position = goal_point
    # Initialise list for solution path, starting at current position
    solution_path = [current_position]

    if algorithm_selected == 'IDS':
        # IDS function returns the complete path without reconstructing
        solution_path = paths + solution_path

    elif algorithm_selected == 'ASTAR':
        # ASTAR function returns the complete path without reconstructing
        paths.insert(0, start_point)
        solution_path = paths

    else:
        # Loops until returned to start point
        while current_position != start_point:
            # Set new position as current, and append to solution path list
            if algorithm_selected == 'BFS':
                current_position = paths[current_position]
            elif algorithm_selected == 'UCS':
                current_position = paths[current_position]['parent']
            solution_path.append(current_position)
        # Reverse the list to find the solution path from start to goal
        solution_path.reverse()

    return solution_path


def path_found_hcs(best_sequence, start_point, goal_point):
    """
    Function specific for HCS solution, to construct the path to reach the goal point
    Returns the solution path that the algorithm found
    """
    # Initialise the current position and solution path
    current_position = start_point
    solution_path = [current_position]

    # Loop through sequence
    for move in best_sequence:
        # Find next position
        next_position = (
            current_position[0] + move[0], current_position[1] + move[1])

        # Check if goal reached
        if next_position == goal_point:
            solution_path.append(next_position)
            break

        # Else, append to path, set next pos and continue looping
        else:
            solution_path.append(next_position)
            current_position = next_position

    # Return the solution path
    return solution_path


def sequence_and_cost(solution):
    """
    Function to loop through the solution to find what the actual moves taken were, & calculate cost
    Returns the solution sequence of moves and the calculated cost of the path found
    """
    solution_sequence = []
    calculated_cost = 0
    # Loop through solution cells
    for x in range(len(solution) - 1):
        # Find what the move was for this step
        solution_move = tuple(
            [solution[x + 1][y] - solution[x][y] for y in range(2)])

        # Lookup dict. move value at index 1, then append to list
        solution_sequence.append(MOVES_DICTIONARY[solution_move][1])
        # Lookup dict. move value at index 0, and add to calculated_cost
        calculated_cost += MOVES_DICTIONARY[solution_move][0]

    return solution_sequence, calculated_cost


def visualise_path(visualisation, discovered, solution=None, random_path=None):
    """
    Function to loop through the solution & replace the path cells with a new value to visualise path taken
    Returns the visualisation of the path taken on the board from start to goal point
    """
    # Define border symbols
    row_edge = '|'
    column_edge = '='
    corners = '+'
    # Equal dimensions so width is both dimensions
    width = len(visualisation)

    # Create new visualisation grid
    bordered_visualisation = [[column_edge] * (width + 2)]
    for row in visualisation:
        bordered_visualisation.append([row_edge] + row + [row_edge])
    bordered_visualisation.append([column_edge] * (width + 2))

    # Write the solution path to visualised board if it exists
    if solution:
        # Loop through steps in solution
        for step in solution:
            cell = bordered_visualisation[step[0] + 1][step[1] + 1]
            if cell != 'S' and cell != 'G':
                # Replace path cell with '@' if not start or goal
                bordered_visualisation[step[0] + 1][step[1] + 1] = '@'

    # Write the initial random path to visualised board if it exists
    if random_path:
        # Loop through steps in random path
        for step in random_path:
            cell = bordered_visualisation[step[0] + 1][step[1] + 1]
            # Replace random path cell with 'x' if not start, goal or solution path '@'
            if cell != 'S' and cell != 'G' and cell != '@':
                bordered_visualisation[step[0] + 1][step[1] + 1] = 'x'

    # Change any of the symbols inside ' ' strings below to try different visualisations
    for r in range(1, len(bordered_visualisation) - 1):
        for c in range(1, len(bordered_visualisation[r]) - 1):

            # Visualise the discovered cells during execution
            if (r - 1, c - 1) in discovered and bordered_visualisation[r][c] == 'R':
                bordered_visualisation[r][c] = '.'

            # Visualise remaining normal ground tiles
            if bordered_visualisation[r][c] == 'R':
                bordered_visualisation[r][c] = ' '

            # Visualise cliffs
            elif bordered_visualisation[r][c] == 'X':
                bordered_visualisation[r][c] = '#'

    # Finally, update corners
    for r in (0, -1):
        for c in (0, -1):
            bordered_visualisation[r][c] = corners

    return bordered_visualisation


def write_output_file(algorithm_selected, solution_sequence=None, calculated_cost=None, explored_states=None, visualisation=None, process_input_time=None, algorithm_time=None):
    """
    Function to write the results found to an output file
    """
    # Create output folder
    output_folder = os.path.join(os.path.dirname(
        input_file_path), f'{algorithm_selected}_output_folder')
    os.makedirs(output_folder, exist_ok=True)

    # Create output filepath
    output_file_path = os.path.join(
        output_folder, f'{input_file_name}_output.txt')

    # Write output file with solution sequence, total path cost, path length, visualisation of path, and any other stats
    if solution_sequence:
        # Set separater value
        sep = '-'
        # create/open output.txt with write priviledge
        with open(output_file_path, 'w') as f:
            # Join all values in solution_sequence with - as separator
            line = sep.join(i for i in solution_sequence)
            f.write(f"{line}\n")

            f.write('\n')

            f.write(f"Path cost: {calculated_cost}\n")

            f.write(f"Path length: {len(solution_sequence)}\n")

            f.write(f"Explored states: {explored_states}\n")

            f.write('\n')

            # New separator
            sep = ' '
            for j in range(len(visualisation)):
                # Write joined line of each row in path visualisation
                line = sep.join(k for k in visualisation[j])
                f.write(f"{line}\n")

            f.write("\nTile Keys\n")

            f.write("\n\t'@'\t-\tPath")
            f.write("\n\t'#'\t-\tCliff")
            f.write("\n\t'.'\t-\tDiscovered")
            f.write("\n\t' '\t-\tNot Discovered")

            f.write('\n')

            f.write(
                f"\nTime taken to process input file: {int(process_input_time)} microseconds")

            f.write(
                f"\nTime taken to execute algorithm: {int(algorithm_time)} microseconds")

        # File closed & location of output displayed to user
        print(f"Path found, output file was written to:\n{output_file_path}\n")

    # Check if no solution found, write 'no path' and the number of explored states
    else:
        with open(output_file_path, 'w') as f:
            # Set separator value
            sep = ' '

            f.write("no path\n")

            f.write(f"Explored states: {explored_states}\n")

            f.write('\n')

            for j in range(len(visualisation)):
                # Write joined line of each row in path visualisation
                line = sep.join(k for k in visualisation[j])
                f.write(f"{line}\n")

            f.write("\nTile Keys\n")

            f.write("\n\t'@'\t-\tPath")
            f.write("\n\t'#'\t-\tCliff")
            f.write("\n\t'.'\t-\tDiscovered")
            f.write("\n\t' '\t-\tNot Discovered")

            f.write('\n')

            f.write(
                f"\nTime taken to process input file: {int(process_input_time)} microseconds")

            f.write(
                f"\nTime taken to execute algorithm: {int(algorithm_time)} microseconds")

        # File closed & location of output displayed to user
        print(
            f"No path found, output file was written to:\n{output_file_path}\n")


def write_output_file_hcs(algorithm_selected, solution_sequence=None, calculated_cost=None, steps_to_end=None, visualisation=None, process_input_time=None, algorithm_time=None):
    """
    Function to write the HCS results found to an output file
    """
    # Create output folder
    output_folder = os.path.join(os.path.dirname(
        input_file_path), 'HCS_output_folder')
    os.makedirs(output_folder, exist_ok=True)

    # Create output filepath (changes based on algorithm selected, this helps us identify the different HCS version output files)
    output_file_path = os.path.join(
        output_folder, f'{input_file_name}_{algorithm_selected}_output.txt')

    # Write output file with solution sequence, total path cost, path length, visualisation of path, and any other stats
    if solution_sequence:
        # Set separater value
        sep = '-'
        # create/open output.txt with write priviledge
        with open(output_file_path, 'w') as f:
            # Join all values in solution_sequence with - as separator
            line = sep.join(i for i in solution_sequence)
            f.write(f"{line}\n")

            f.write('\n')

            f.write(f"Path cost: {calculated_cost}\n")

            f.write(f"Path length: {len(solution_sequence)}\n")

            f.write(f"Steps to End: {steps_to_end}\n")

            f.write('\n')

            # New separator
            sep = ' '

            # Final HCS solution visualisation
            f.write(f"Final {algorithm_selected} HCS solution visualisation\n")

            for j in range(len(visualisation)):
                # Write joined line of each row in path visualisation
                line = sep.join(k for k in visualisation[j])
                f.write(f"{line}\n")

            f.write("\nTile Keys\n")

            f.write("\n\t'@'\t-\tPath")
            f.write("\n\t'#'\t-\tCliff")
            f.write("\n\t'.'\t-\tDiscovered")
            f.write("\n\t' '\t-\tNot Discovered")
            f.write("\n\t'x'\t-\tInitial Random Path")

            f.write('\n')

            f.write(
                f"\nTime taken to process input file: {int(process_input_time)} microseconds")

            f.write(
                f"\nTime taken to execute algorithm: {int(algorithm_time)} microseconds")

        # File closed & location of output displayed to user
        print(f"Output file was written to:\n{output_file_path}\n")

###################
### Driver code ###
###################


# Process input file and time to process
# Variable processed_input_results contains the following fields: algorithm_selected, board_size_N, board, start_point, goal_point
processed_input_results, process_input_time = timer(process_input_file)()
# Define the results for later use
algorithm_selected, board_size_N, board, start_point, goal_point = processed_input_results[:]

# Match and execute the correct algorithm, return the result and execution time as a tuple from timer decorator
# Variable execution_results contains the following fields: solution, explored_states
match algorithm_selected:
    case 'BFS':
        execution_results, algorithm_time = timer(
            execute_bfs)(*processed_input_results)

    case 'UCS':
        execution_results, algorithm_time = timer(
            execute_ucs)(*processed_input_results)

    case 'IDS':
        execution_results, algorithm_time = timer(
            execute_ids)(*processed_input_results)

    case 'ASTAR':
        execution_results, algorithm_time = timer(
            execute_astar)(*processed_input_results)

    case 'HCS':
        # Find a solution from executing BFS search algorithm
        bfs_execution_results, bfs_algorithm_time = timer(execute_bfs)(
            'BFS', board_size_N, board, start_point, goal_point)
        bfs_solution, bfs_explored_states, bfs_discovered = bfs_execution_results[:]

        # Check if solution is not None
        if bfs_solution is None:
            # Create board visualisation with no path
            visualisation = visualise_path(board, bfs_discovered)
            # Write output with no solution
            write_output_file(algorithm_selected, solution_sequence=None, explored_states=bfs_explored_states,
                              visualisation=visualisation, process_input_time=process_input_time, algorithm_time=bfs_algorithm_time)
            sys.exit()
        else:
            pass

        # Initialise random_path and loop to ensure we find a sequence of moves that can reach the goal point
        random_path = []
        solution_length_M = len(bfs_solution) - 1
        while len(random_path) < len(bfs_solution):
            random_sequence, random_path = get_random_start(
                solution_length_M, board_size_N, board, start_point)

        # Let user select version of HCS
        selected_version = versions_window()

        # Match selected version to execute the correct algorithm
        match selected_version:
            case 'Greedy':
                # Execute Greedy HCS search algorithm, parse the processed input, random sequence and bfs solution
                execution_results, algorithm_time = timer(execute_hcs_greedy)(
                    board_size_N, board, start_point, goal_point, random_sequence, bfs_solution)

            case 'Random_Restart':
                # Execute Random Restart HCS search algorithm, parse the processed input, random sequence and bfs solution
                execution_results, algorithm_time = timer(execute_hcs_random_restart)(
                    board_size_N, board, start_point, goal_point, bfs_solution)

            case 'Randomised':
                # Execute Randomised HCS search algorithm and parse, parse the processed input, random sequence and bfs solution
                execution_results, algorithm_time = timer(execute_hcs_randomised)(
                    board_size_N, board, start_point, goal_point, random_sequence, bfs_solution)

            # Handle version selection error
            case _:
                handle_error(
                    'Selected version - User may have closed the version selection window...')

# Process execution results and write to output file
# Check if HCS, as we execute some of the remaining functions differently for HCS
if algorithm_selected == 'HCS':
    # Define the solution and steps to end value
    solution, steps_to_end, discovered = execution_results[:]

    # Check if the random sequence started with a path to goal point, and was also a local min/max (Steps to end should be 1)
    if random_path[-1] == goal_point and steps_to_end <= 1:
        # Let user know what happened, and set solution to the random path so we can see visually from the output file
        print("Random sequence was initiated with a path to the goal point, but became stuck in a local min/max and could not optimise...")
        solution = random_path

    # Check if solution is not None
    if solution:
        # Find all values we wish to write to the output file
        # We find solution sequence and calculate path cost at the same time, as they would be almost identical functions
        solution_sequence, calculated_cost = sequence_and_cost(solution)
        # Create board visualisation of the random path and final path
        visualisation = visualise_path(
            board, discovered, solution=solution, random_path=random_path)
        # Write all to output.txt file
        write_output_file_hcs(selected_version, solution_sequence, calculated_cost,
                              steps_to_end, visualisation, process_input_time, algorithm_time)

    else:
        print(discovered)
        # Create board visualisation of the random path but no final solution
        visualisation = visualise_path(
            board, discovered, solution=None, random_path=random_path)
        # Write output with no solution
        write_output_file_hcs(selected_version, steps_to_end=steps_to_end, visualisation=visualisation,
                              process_input_time=process_input_time, algorithm_time=algorithm_time)

# The rest of the algorithms execution results are processed the similarly to each other
else:
    # Define the solution and explored states value
    solution, explored_states, discovered = execution_results[:]

    # Check if solution is not None
    if solution:
        # Find all values we wish to write to the output file
        # We find solution sequence and calculate path cost at the same time, as they would be almost identical functions
        solution_sequence, calculated_cost = sequence_and_cost(solution)
        # Create board visualisation of path
        visualisation = visualise_path(board, discovered, solution)
        # Write all to output.txt file
        write_output_file(algorithm_selected, solution_sequence, calculated_cost,
                          explored_states, visualisation, process_input_time, algorithm_time)

    else:
        # Create board visualisation with no path
        visualisation = visualise_path(board, discovered)
        # Write output with no solution
        write_output_file(algorithm_selected, solution_sequence=None, explored_states=explored_states,
                          visualisation=visualisation, process_input_time=process_input_time, algorithm_time=algorithm_time)
