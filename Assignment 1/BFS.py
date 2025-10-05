from collections import deque

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [initial_state])])
    visited = set()

    while queue:
        state, path = queue.popleft()

        if state == goal_state:
            return path

        visited.add(state)

        # Example expansion logic (generic placeholder)
        next_states = [] 
        for next_state in next_states:
            if next_state not in visited:
                queue.append((next_state, path + [next_state]))

    return None
