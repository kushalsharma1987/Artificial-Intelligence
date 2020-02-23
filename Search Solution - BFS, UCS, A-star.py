from math import sqrt
import time
import heapq


# Function to find the adjacency list of given node.
def adjacency_list(state):
    adj_dict = {}
    adj_line_list = []
    adj_diag_list = []
    coord = state.split(" ")
    year = coord[0]
    x = int(coord[1])
    y = int(coord[2])
    vertices = "{0} {1} {2}"

    # Line adjacent
    if y + 1 <= max_y:
        adj_line_list.append(vertices.format(year, x, y + 1))  # N
    if x + 1 <= max_x:
        adj_line_list.append(vertices.format(year, x + 1, y))  # E
    if y - 1 >= 0:
        adj_line_list.append(vertices.format(year, x, y - 1))  # S
    if x - 1 >= 0:
        adj_line_list.append(vertices.format(year, x - 1, y))  # W

    # Diagonal adjacent
    if x + 1 <= max_x and y + 1 <= max_y:
        adj_diag_list.append(vertices.format(year, x + 1, y + 1))  # NE
    if x + 1 <= max_x and y - 1 >= 0:
        adj_diag_list.append(vertices.format(year, x + 1, y - 1))  # SE
    if x - 1 >= 0 and y - 1 >= 0:
        adj_diag_list.append(vertices.format(year, x - 1, y - 1))  # SW
    if x - 1 >= 0 and y + 1 <= max_y:
        adj_diag_list.append(vertices.format(year, x - 1, y + 1))  # NW

    # Combine both Line adjacent list and Diagonal adjacent list
    adj_list_comb = [adj_line_list, adj_diag_list]
    adj_dict[state] = adj_list_comb
    return adj_dict


# Function to jump to another world and search in that world space at given jaunt state
def explore_jaunt(entry_state):
    entry_coords = entry_state.split(" ")
    entry_year = int(entry_coords[0])
    exit_year = int(jaunt_loc[entry_state])
    exit_state = str(exit_year) + " " + entry_coords[1] + " " + entry_coords[2]
    # print("exit_state:", exit_state)
    return exit_state


def jaunt_cost(entry_state):
    entry_coords = entry_state.split(" ")
    entry_year = int(entry_coords[0])
    exit_year = int(jaunt_loc[entry_state])
    j_cost = abs(exit_year - entry_year)
    return j_cost


def future_cost(e):
    current_coord = e.split(" ")
    current_year = int(current_coord[0])
    current_x = int(current_coord[1])
    current_y = int(current_coord[2])
    # Euclidean distance between two points in 3D
    eucl_distance = sqrt(((target_x - current_x) ** 2) + ((target_y - current_y) ** 2)
                         + ((target_year - current_year) ** 2))
    # diagonal distance
    # diag_distance = max(abs(current_x - target_x), abs(current_y - target_y))
    # manhattan distance
    # manhat_distance = abs(current_x - target_x) + abs(current_y - target_y)
    # Estimated future cost = Euclidean Distance + Jump cost from one world to another
    est_future_cost = eucl_distance
    # est_future_cost = eucl_distance + abs(target_year - current_year)
    # est_future_cost = diag_distance * 14 + abs(target_year - current_year)
    # est_future_cost = manhat_distance * 10 + abs(target_year - current_year)
    return int(est_future_cost)


# Function to perform sorting for UCS or A* Search based on heuristic.


# Function to perform Breadth First Search
def bfs(initial_state):
    # i = 1
    a = 0
    frontier = [initial_state]
    while frontier:
        global iterations
        iterations = iterations + 1
        next_row = []
        for u in frontier:
            if u in jaunt_loc:
                # if explore_jaunt(u):
                #     break
                j_state = explore_jaunt(u)
                j_cost = jaunt_cost(u)
                if j_state not in visited:
                    visited[j_state] = j_cost
                    parent[j_state] = u
                    cost[j_state] = cost[u] + visited[v]
                    next_row.append(j_state)

            adj_list_comb = adjacency_list(u)
            # Since Line adjacent and Diagonal adjacent have same step cost, combine them in one.
            adj_list = adj_list_comb[u][0] + adj_list_comb[u][1]
            for v in adj_list:
                if v not in visited:
                    visited[v] = 1
                    parent[v] = u
                    cost[v] = cost[u] + visited[v]
                    next_row.append(v)
        # If target state is found, stop the algorithm and exit
        if target_state in next_row:
            break
        global expanded_list
        expanded_list = expanded_list + len(next_row)
        frontier = next_row
        if a < len(frontier):
            a = len(frontier)
            # frontier = frontier + next_row
            # Sort the frontier list based on sorting criteria to make a new frontier list to explore.
            # frontier.sort(key=sorting)
    print("Max Frontier Length:", a)


# Function to perform Uniform Cost Search(UCS) or A*. Only difference is in Sorting criteria
def ucs(initial_state):
    # frontier = [initial_state]
    a = 0
    cost[initial_state] = future_cost(initial_state)
    frontier = []
    heapq.heappush(frontier, (cost[initial_state], initial_state))
    while frontier:
        global iterations
        iterations = iterations + 1
        # Pick first item from the list
        # u = frontier.pop(0)
        heap_tuple = heapq.heappop(frontier)
        u = heap_tuple[1]
        # If current state is target state, we stop because we found path with minimal cost.
        if u == target_state:
            break
        next_row = []
        # Jump to another world if there is jaunt at this node
        if u in jaunt_loc:
            j_state = explore_jaunt(u)
            j_cost = jaunt_cost(u)
            if j_state in visited:
                if cost[j_state] > cost[u] + j_cost:
                    visited.pop(j_state)
                    parent.pop(j_state)
                    cost.pop(j_state)
                    # if j_state in frontier:
                    #     frontier.remove(j_state)
                    visited[j_state] = j_cost
                    parent[j_state] = u
                    cost[j_state] = cost[u] + visited[j_state]
                    next_row.append(j_state)
            else:
                visited[j_state] = j_cost
                parent[j_state] = u
                cost[j_state] = cost[u] + visited[j_state]
                next_row.append(j_state)

        adj_list_comb = adjacency_list(u)
        adj_line_list = adj_list_comb[u][0]
        adj_diag_list = adj_list_comb[u][1]
        # Expand the Line adjacency list of a node
        for v in adj_line_list:
            if v in visited:
                # If a node is already visited and the path cost of the already visited node is more than the cost
                # derived from new path, remove the already visited node and add the newly explored node.
                if cost[v] > cost[u] + 10:
                    visited.pop(v)
                    parent.pop(v)
                    cost.pop(v)
                    # if v in frontier:
                    #     frontier.remove(v)
                else:
                    continue
            visited[v] = 10
            parent[v] = u
            cost[v] = cost[u] + visited[v]
            next_row.append(v)
        # Expand the diagonal adjacency list
        for v in adj_diag_list:
            if v in visited:
                # If a node is already visited and the path cost of the already visited node is more than the cost
                # derived from new path, remove the already visited node and add the newly explored node.
                if cost[v] > cost[u] + 14:
                    visited.pop(v)
                    parent.pop(v)
                    cost.pop(v)
                    # if v in frontier:
                    #     frontier.remove(v)
                else:
                    continue
            visited[v] = 14
            parent[v] = u
            cost[v] = cost[u] + visited[v]
            next_row.append(v)
        global expanded_list
        expanded_list = expanded_list + len(next_row)
        # Add the next_row list into existing frontier Heap
        for i in next_row:
            heapq.heappush(frontier, (cost[i], i))
        if a < len(frontier):
            a = len(frontier)

        # frontier = frontier + next_row
        # Sort the frontier list based on sorting criteria to make a new frontier list to explore.
        # frontier.sort(key=sorting)
    print("Max Frontier Length:", a)


# Function to perform A* Search.
def astar(initial_state):
    a = 0
    # frontier = [initial_state]
    cost[initial_state] = future_cost(initial_state)
    frontier = []
    heapq.heappush(frontier, (cost[initial_state], initial_state))
    # heappush(frontier, Frontier(initial_state, future_cost(initial_state)))
    # heapq.heapify(frontier)
    while frontier:
        global iterations
        iterations = iterations + 1
        # Pick first item from the list
        # u = frontier.pop(0)
        heap_tuple = heapq.heappop(frontier)
        u = heap_tuple[1]
        # print("State", u, "Value", heap_tuple[0])
        # u = heapq.heappop(frontier)
        # If current state is target state, we stop because we found path with minimal cost.
        if u == target_state:
            break
        next_row = []
        # Check if there is jaunt at this node, calculate the cost and put in list
        if u in jaunt_loc:
            j_state = explore_jaunt(u)
            j_cost = jaunt_cost(u)
            future_goal_cost = future_cost(j_state)
            if j_state in visited:
                if cost[j_state] > cost[u] + j_cost + future_goal_cost:
                    visited.pop(j_state)
                    parent.pop(j_state)
                    cost.pop(j_state)
                    # if j_state in frontier:
                    #     frontier.remove(j_state)
                    visited[j_state] = j_cost
                    parent[j_state] = u
                    cost[j_state] = cost[u] + visited[j_state] + future_goal_cost
                    next_row.append(j_state)
            else:
                visited[j_state] = j_cost
                parent[j_state] = u
                cost[j_state] = cost[u] + visited[j_state] + future_goal_cost
                next_row.append(j_state)

        adj_list_comb = adjacency_list(u)
        adj_line_list = adj_list_comb[u][0]
        adj_diag_list = adj_list_comb[u][1]

        # Expand the Line adjacency list of a node
        for v in adj_line_list:
            future_goal_cost = future_cost(v)
            if v in visited:
                # If a node is already visited and the path cost of the already visited node is more than the cost
                # derived from new path, remove the already visited node and add the newly explored node.
                if cost[v] > cost[u] + 10 + future_goal_cost:
                    visited.pop(v)
                    parent.pop(v)
                    cost.pop(v)
                    # if v in frontier:
                    #     frontier.remove(v)
                else:
                    continue
            visited[v] = 10
            parent[v] = u
            cost[v] = cost[u] + visited[v] + future_goal_cost
            next_row.append(v)
        # Expand the diagonal adjacency list
        for v in adj_diag_list:
            future_goal_cost = future_cost(v)
            if v in visited:
                # If a node is already visited and the path cost of the already visited node is more than the cost
                # derived from new path, remove the already visited node and add the newly explored node.
                if cost[v] > cost[u] + 14 + future_goal_cost:
                    visited.pop(v)
                    parent.pop(v)
                    cost.pop(v)
                    # if v in frontier:
                    #     frontier.remove(v)
                else:
                    continue
            visited[v] = 14
            parent[v] = u
            cost[v] = cost[u] + visited[v] + future_goal_cost
            next_row.append(v)
        global expanded_list
        expanded_list = expanded_list + len(next_row)
        # Add the next_row list into existing frontier heap
        for i in next_row:
            heapq.heappush(frontier, (cost[i], i))
        if a < len(frontier):
            a = len(frontier)
        # frontier = frontier + next_row
        # heapq.heapify(frontier)
        # Sort the frontier list based on sorting criteria to make a new frontier list to explore.
        # frontier.sort(key=sorting)
    print("Max Frontier Length:", a)


# Function to Search the space with given initial state based on algorithm from input.
def search(initial_state):
    if algo == "BFS":
        bfs(initial_state)
    elif algo == "UCS":
        ucs(initial_state)
    elif algo == "A*":
        astar(initial_state)


# Open the input.txt and read the given parameters into respective variables.
start_time = time.time()
fr = open("input.txt", "r")
algo = fr.readline().replace("\n", "")
dimension = fr.readline().replace("\n", "").split(" ")
max_x = int(dimension[0])
max_y = int(dimension[1])
initial_state = fr.readline().replace("\n", "")
target_state = fr.readline().replace("\n", "")
target_coord = target_state.split(" ")
target_year = int(target_coord[0])
target_x = int(target_coord[1])
target_y = int(target_coord[2])
num_jaunts = int(fr.readline().replace("\n", ""))
jaunt_loc = {}
for j in range(num_jaunts):
    jaunt = fr.readline().replace("\n", "")
    b = jaunt.split(" ")
    c = b[0] + " " + b[1] + " " + b[2]
    d = b[3]
    jaunt_loc[c] = d

# Work variables
iterations = 0
expanded_list = 0
visited = {initial_state: 0}
parent = {initial_state: "None"}
cost = {initial_state: 0}

# input.txt File close
fr.close()
print("Algorithm:", algo)
# Call Search function
search(initial_state)

# After algorithm completion, open output.txt for writing the output
fw = open("output.txt", "w")

# if target not found, write "FAIL"
if target_state not in visited:
    print("FAIL")
    fw.write("FAIL\n")
    fw.close()
    exit()

# if target found, write the respective output.
s = target_state
path = []
path_cost = 0
step_cost = {}
while s != initial_state:
    path.append(s)
    path_cost = path_cost + visited[s]
    step_cost[s] = cost[s]
    s = parent[s]
path.append(initial_state)
step_cost[initial_state] = 0
path.reverse()
# print("Step Cost: ", step_cost)
print("Path Cost: ", path_cost)
fw.write(str(path_cost) + "\n")  # Total path cost
print("Steps: ", len(path))
fw.write(str(len(path)) + "\n")  # Total steps in the path
print("Path: ", path)
for x in path:
    fw.write(x + " " + str(visited[x]) + "\n")  # Step by step along with cost
print("Iterations:", iterations)
print("Expanded_list:", expanded_list)
print("Time:", time.time() - start_time)

fw.close()
