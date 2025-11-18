import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from heapq import heappush, heappop
import random


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


# ---------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------
def greedy_bfs_steps(grid, start, goal):

    heuristic = lambda p: ((p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2)**0.5

    open_set = []
    heappush(open_set, (heuristic(start), start))

    closed = set()
    came_from = {}

    while open_set:
        _, current = heappop(open_set)
        if current in closed:
            continue

        closed.add(current)

        # reconstruct partial path for visualization
        path = []
        node = current
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        path.reverse()

        # yield step
        yield current, closed.copy(), [p for _, p in open_set], path

        if current == goal:
            return  # path found – animation will end

        # explore neighbors
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if grid[nr, nc] == 0 and (nr, nc) not in closed:
                heappush(open_set, (heuristic((nr, nc)), (nr, nc)))
                if (nr, nc) not in came_from:
                    came_from[(nr, nc)] = current


# ---------------------------------------------------------
# Animation wrapper (like your Dijkstra animation)
# ---------------------------------------------------------
def animate_greedy_bfs(grid, start, goal, interval=60):
    fig, ax = plt.subplots()
    grid_rgb = np.ones((*grid.shape, 3))

    # Colors
    closed_color = np.array([1, 1, 0])          # yellow
    open_color   = np.array([0.3, 0.8, 1])      # blue
    path_color   = np.array([0, 1, 0.5])        # teal
    start_color  = np.array([0, 1, 0])          # green
    goal_color   = np.array([1, 0, 0])          # red
    current_color = np.array([1, 0.5, 0])       # orange

    # Initialize iterator
    steps = greedy_bfs_steps(grid, start, goal)

    # initial image
    img = ax.imshow(grid_rgb, interpolation="nearest")
    ax.set_title("Greedy Best-First Search")

    def update(_):
        result = next(steps, None)
        if not result:
            return img

        current, closed, open_, path = result

        # reset base grid
        grid_rgb[:] = 1
        grid_rgb[grid == 1] = 0  # obstacles -> black

        # Closed set
        for (x, y) in closed:
            grid_rgb[x, y] = closed_color

        # Open set
        for (x, y) in open_:
            grid_rgb[x, y] = open_color

        # Path
        for (x, y) in path:
            grid_rgb[x, y] = path_color

        # Start & Goal
        grid_rgb[start] = start_color
        grid_rgb[goal] = goal_color

        # Current node
        if current:
            grid_rgb[current] = current_color

        img.set_data(grid_rgb)
        return img

    _ = animation.FuncAnimation(
        fig,
        update,
        interval=interval,
        blit=False,
        cache_frame_data=False
    )

    plt.show()

# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
if __name__ == "__main__":
    size = int(input("Enter grid size (e.g., 20): "))
    while size < 5:
        size = int(input('Size cannot be lower than 5: '))
    # Generate random maze
    grid = generate_grid(size)

    start = (2, 2)
    goal = (size - 2, size - 2)

    # Ensure start and goal are free
    grid[start] = 0
    grid[goal] = 0

    print(f"Grid size with border: {grid.shape}")
    print(f"Start: {start}, Goal: {goal}")

    animate_greedy_bfs(grid, start, goal)

