import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# -------------------- Dijkstra Node --------------------
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # cost from start (distance)
    
    def __lt__(self, other):
        return self.g < other.g


# -------------------- Dijkstra Generator (yields each step) --------------------
def dijkstra_steps(grid, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start)
    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)
        closed_set.add(current.position)

        # Yield visualization step
        yield current.position, list(closed_set), [n.position for n in open_list], None

        if current.position == goal:
            # reconstruct path
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            yield None, list(closed_set), [n.position for n in open_list], path[::-1]
            return

        x, y = current.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for nx, ny in neighbors:
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx][ny] == 0:
                neighbor = Node((nx, ny), current)
                if neighbor.position in closed_set:
                    continue

                neighbor.g = current.g + 1  # uniform cost

                # If a node with lower cost already exists in open_list, skip
                if any(n.position == neighbor.position and n.g <= neighbor.g for n in open_list):
                    continue

                heapq.heappush(open_list, neighbor)

    yield None, list(closed_set), [n.position for n in open_list], None


# -------------------- Maze Generator --------------------
def generate_grid(size, obstacle_prob=0.25):
    grid = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i][j] = 1
    return grid


# -------------------- Animation --------------------
def animate_dijkstra(grid, start, goal, interval=80):
    fig, ax = plt.subplots()
    n = grid.shape[0]
    grid_rgb = np.ones((n, n, 3))
    grid_rgb[grid == 1] = [0, 0, 0]  # black obstacles
    img = ax.imshow(grid_rgb, interpolation="nearest")
    ax.set_title("Dijkstra Pathfinding Animation")
    ax.axis("off")

    steps = dijkstra_steps(grid, start, goal)
    closed_color = np.array([1, 1, 0])        # yellow
    open_color   = np.array([0.3, 0.8, 1])    # blue
    path_color   = np.array([0, 1, 0.5])      # teal
    start_color  = np.array([0, 1, 0])        # green
    goal_color   = np.array([1, 0, 0])        # red

    def update(frame):
        result = next(steps, None)
        if not result:
            return img

        current, closed, open_, path = result

        grid_rgb[:] = 1
        grid_rgb[grid == 1] = 0  # obstacles black

        for (x, y) in closed:
            grid_rgb[x, y] = closed_color
        for (x, y) in open_:
            grid_rgb[x, y] = open_color
        if path:
            for (x, y) in path:
                grid_rgb[x, y] = path_color

        grid_rgb[start] = start_color
        grid_rgb[goal] = goal_color

        if current:
            grid_rgb[current] = [1, 0.5, 0]  # current node = orange

        img.set_data(grid_rgb)
        return img

    _ = animation.FuncAnimation(fig, update, interval=interval, blit=False, cache_frame_data=False)
    plt.show()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    size = int(input("Enter grid size (e.g. 20 for 20x20): "))

    while size < 5:
        size = int(input('Size cannot be lower than 5: '))
    # Generate random maze
    grid = generate_grid(size)

    # Add 1-cell border
    grid = np.pad(grid, pad_width=1, mode='constant', constant_values=1)

    #Start and Goal on opposite corners
    start = (2, 2)
    goal = (grid.shape[0] - 3, grid.shape[1] - 3)

    # Ensure start and goal are free
    grid[start] = 0
    grid[goal] = 0

    print(f"Grid size with border: {grid.shape}")
    print(f"Start: {start}, Goal: {goal}")
    
    animate_dijkstra(grid, start, goal)
