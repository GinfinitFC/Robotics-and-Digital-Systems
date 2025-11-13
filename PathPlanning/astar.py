import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# -------------------- A* Node Definition --------------------
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -------------------- A* Generator (yields steps) --------------------
def astar_steps(grid, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)
        closed_set.add(current.position)

        # Yield current state for animation
        yield current.position, list(closed_set), [n.position for n in open_list], None

        if current.position == goal_node.position:
            # Reconstruct path
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            yield None, list(closed_set), [n.position for n in open_list], path[::-1]
            return

        x, y = current.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for nx, ny in neighbors:
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                neighbor = Node((nx, ny), current)
                if neighbor.position in closed_set:
                    continue

                neighbor.g = current.g + 1
                neighbor.h = heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h

                if any(n.position == neighbor.position and n.f <= neighbor.f for n in open_list):
                    continue

                heapq.heappush(open_list, neighbor)

    yield None, list(closed_set), [n.position for n in open_list], None


# -------------------- Random Maze Generator --------------------
def generate_grid(size, obstacle_prob=0.20):
    border = np.ones((size+2,size+2), dtype=int)
    grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i][j] = 1
    
    border[1:-1, 1:-1] = grid
    return border


def random_free_cell(grid):
    free = list(zip(*np.where(grid == 0)))
    return random.choice(free) if free else None


# -------------------- Visualization --------------------
def animate_astar(grid, start, goal, interval=80):
    fig, ax = plt.subplots()
    n = len(grid)
    grid_rgb = np.ones((n, n, 3))
    grid_rgb[grid == 1] = [0, 0, 0]  # obstacles = black
    img = ax.imshow(grid_rgb, interpolation="nearest")
    ax.set_title("A* Pathfinding Animation")
    ax.axis("off")

    steps = astar_steps(grid, start, goal)
                            #R  G  B
    closed_color = np.array([1, 1, 0])  # yellow
    open_color = np.array([0.3, 0.8, 1])  # blue
    path_color = np.array([0, 1, 0.5])  # teal
    start_color = np.array([0, 1, 0])  # green
    goal_color = np.array([1, 0, 0])  # red

    def update(frame):
        result = next(steps, None)
        if not result:
            return img

        current, closed, open_, path = result
        grid_rgb[:] = 1  # reset
        grid_rgb[grid == 1] = [0, 0, 0]

        for (x, y) in closed:
            grid_rgb[x, y] = closed_color
        for (x, y) in open_:
            grid_rgb[x, y] = open_color
        if path:
            for (x, y) in path:
                grid_rgb[x, y] = path_color
        if start:
            grid_rgb[start] = start_color
        if goal:
            grid_rgb[goal] = goal_color
        if current:
            grid_rgb[current] = [1, 0.5, 0]  # orange (current node)
        img.set_data(grid_rgb)
        return img

    ani = animation.FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    plt.show()


# -------------------- MAIN PROGRAM --------------------
if __name__ == "__main__":
    size = int(input("Enter grid size (e.g. 20 for 20x20): "))
    while size<5:
        size = int(input("size cannot be lower than 5: "))
    grid = generate_grid(size)

    #this chunk of code makes the start and goal random
    '''
    start = random_free_cell(grid)
    goal = random_free_cell(grid)
    while goal == start:
        goal = random_free_cell(grid)
    '''

    # Force start and goal in opposite corners
    start = (2, 2)
    goal = (size - 1, size - 1)

    # Ensure start and goal are free
    grid[start] = 0
    grid[goal] = 0

    print(f"Start: {start}, Goal: {goal}")
    animate_astar(grid, start, goal, interval=80)
