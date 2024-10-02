import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

RING_RADIUS = 5.0
NUM_ACTIONS = 40
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
NUM_GENERATIONS = 100
FRAMES_PER_ACTION = 30 
ROBOT_RADIUS = 0.25

class Robot:
    def __init__(self):
        self.position = np.array([0.0, 0.0])  
        self.velocity = np.array([0.0, 0.0])  
        self.radius = ROBOT_RADIUS  
        self.reset()

    def reset(self):
        """Reset the robot to the center of the ring."""
        self.position = np.array([0.0, 0.0])

    def move(self, action):
        """Move the robot based on the action."""
        self.velocity = action
        self.position += self.velocity

    def is_in_ring(self):
        """Check if the robot is still within the ring."""
        return np.linalg.norm(self.position) <= RING_RADIUS

def handle_collision(robot1, robot2):
    """Handle collision between two robots."""
    distance_between = np.linalg.norm(robot1.position - robot2.position)
    min_distance = robot1.radius + robot2.radius

    if distance_between < min_distance:  
        
        overlap = min_distance - distance_between
        
        direction = (robot2.position - robot1.position) / distance_between

        robot1.position -= direction * overlap / 2
        robot2.position += direction * overlap / 2

def create_initial_population():
    """Create a population of random action sequences for each robot."""
    population = []
    for _ in range(POPULATION_SIZE):
        actions_robot_1 = [np.random.uniform(-1, 1, 2) for _ in range(NUM_ACTIONS)]
        actions_robot_2 = [np.random.uniform(-1, 1, 2) for _ in range(NUM_ACTIONS)]
        population.append((actions_robot_1, actions_robot_2))
    return population

def simulate_battle(robot1, robot2, actions1, actions2):
    """Simulate a battle between two robots based on their action sequences."""
    robot1.reset()
    robot2.reset()

    for i in range(NUM_ACTIONS):
        
        robot1.move(actions1[i])
        robot2.move(actions2[i])
        
        handle_collision(robot1, robot2)
        
        if not robot1.is_in_ring():
            return 2  
        if not robot2.is_in_ring():
            return 1  
    return 0

def fitness_function(actions_robot_1, actions_robot_2):
    """Calculate fitness for each pair of robots."""
    robot1 = Robot()
    robot2 = Robot()

    proximity_bonus = 0
    push_bonus = 0  
    avoidance_penalty = 0
    edge_push_bonus = 0  
    center_stay_bonus_robot1 = 0
    center_stay_bonus_robot2 = 0

    for action in range(NUM_ACTIONS):
        
        robot1.move(actions_robot_1[action])
        robot2.move(actions_robot_2[action])

        
        handle_collision(robot1, robot2)

        
        distance_between = np.linalg.norm(robot1.position - robot2.position)

        
        if distance_between < 2.0:  
            proximity_bonus += (2.0 - distance_between) * 4  

            push_strength = (2.0 - distance_between) * 1.5  
            push_direction = (robot2.position - robot1.position) / distance_between  
            robot2.position += push_strength * push_direction  
            robot1.position -= push_strength * push_direction  

            push_bonus += push_strength
            
            distance_to_edge_robot2 = RING_RADIUS - np.linalg.norm(robot2.position)
            if distance_to_edge_robot2 < 1.0:  
                edge_push_bonus += (1.0 - distance_to_edge_robot2) * 5  
        
        if distance_between > 3.0:  
            avoidance_penalty += (distance_between - 3.0) * 2  
        
        center_stay_bonus_robot1 += max(0, RING_RADIUS - np.linalg.norm(robot1.position)) * 0.5
        center_stay_bonus_robot2 += max(0, RING_RADIUS - np.linalg.norm(robot2.position)) * 0.5

        if not robot1.is_in_ring():
            return 0.0  
        if not robot2.is_in_ring():
            return 1.0 + proximity_bonus + push_bonus + edge_push_bonus - avoidance_penalty + center_stay_bonus_robot2

    
    fitness_robot1 = proximity_bonus + push_bonus + edge_push_bonus - avoidance_penalty + center_stay_bonus_robot1
    fitness_robot2 = proximity_bonus + push_bonus + edge_push_bonus - avoidance_penalty + center_stay_bonus_robot2

    
    return fitness_robot1 if fitness_robot1 > fitness_robot2 else fitness_robot2

def selection(population, fitness_scores):
    """Select parents using tournament selection."""
    selected = []
    for _ in range(POPULATION_SIZE):
        i1, i2 = random.sample(range(POPULATION_SIZE), 2)
        if fitness_scores[i1] > fitness_scores[i2]:
            selected.append(population[i1])
        else:
            selected.append(population[i2])
    return selected

def crossover(parent1, parent2):
    """Perform crossover between two parents to produce offspring."""
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_ACTIONS - 1)
        child1 = (parent1[0][:point] + parent2[0][point:], parent1[1][:point] + parent2[1][point:])
        child2 = (parent2[0][:point] + parent1[0][point:], parent2[1][:point] + parent1[1][point:])
        return child1, child2
    return parent1, parent2

def mutate(individual):
    """Mutate an individual's genes with a small probability."""
    for i in range(NUM_ACTIONS):
        if random.random() < MUTATION_RATE:
            individual[0][i] = np.random.uniform(-1, 1, 2)
        if random.random() < MUTATION_RATE:
            individual[1][i] = np.random.uniform(-1, 1, 2)
    return individual

def genetic_algorithm():
    """Run the genetic algorithm to evolve robot strategies."""
    population = create_initial_population()  

    for generation in range(NUM_GENERATIONS):
        print(f"\n==> Starting Generation {generation + 1}")
        
        fitness_scores = [fitness_function(ind[0], ind[1]) for ind in population]

        best_fitness = max(fitness_scores)
        print(f"==> End of Generation {generation + 1}: Best fitness: {best_fitness}")
        
        selected_population = selection(population, fitness_scores)
        
        next_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = next_population
        
        best_index = fitness_scores.index(max(fitness_scores))
        best_actions_robot_1, best_actions_robot_2 = population[best_index]

        yield best_actions_robot_1, best_actions_robot_2

def plot_simulation():
    # Create figure and axes for plotting
    fig, ax = plt.subplots()
    ax.set_xlim(-RING_RADIUS, RING_RADIUS)
    ax.set_ylim(-RING_RADIUS, RING_RADIUS)
    ring = plt.Circle((0, 0), RING_RADIUS, fill=False, color='black', lw=2)
    ax.add_artist(ring)
    
    # Create points for the two robots
    robot1_dot, = ax.plot([], [], 'ro', label="Robot 1", markersize=10)
    robot2_dot, = ax.plot([], [], 'bo', label="Robot 2", markersize=10)

    robot1 = Robot()
    robot2 = Robot()

    # Variables to store action sequences and interpolated step counts
    actions_robot_1, actions_robot_2 = [], []
    step = 0
    action_index = 0

    # Initialize the genetic algorithm generator only ONCE
    ga_generator = genetic_algorithm()  # This ensures it's persistent across frames

    def init():
        # Initialize the dots in the plot
        robot1_dot.set_data([], [])
        robot2_dot.set_data([], [])
        return robot1_dot, robot2_dot

    def update(frame):
        nonlocal step, action_index, actions_robot_1, actions_robot_2

        # If all actions for the current generation are done or haven't been initialized
        if action_index >= NUM_ACTIONS or len(actions_robot_1) == 0 or len(actions_robot_2) == 0:
            try:
                # Fetch the next best action sequences
                actions_robot_1, actions_robot_2 = next(ga_generator)

                # Ensure action_index is reset when a new generation begins
                action_index = 0
                step = 0

            except StopIteration:
                print("Genetic algorithm completed all generations.")
                ani.event_source.stop()
                return robot1_dot, robot2_dot  # Always return the artists

        # Ensure that action_index does not exceed the length of action sequences
        if action_index < NUM_ACTIONS:
            # Move the robots smoothly (incremental moves)
            robot1.velocity = actions_robot_1[action_index] / FRAMES_PER_ACTION
            robot2.velocity = actions_robot_2[action_index] / FRAMES_PER_ACTION
            robot1.move(robot1.velocity)
            robot2.move(robot2.velocity)

            # Handle collision if robots are close
            handle_collision(robot1, robot2)

            # Update the plot with their positions
            robot1_dot.set_data([robot1.position[0]], [robot1.position[1]])
            robot2_dot.set_data([robot2.position[0]], [robot2.position[1]])

            # Check if a robot has left the ring
            if not robot1.is_in_ring():
                robot1.reset()
                robot2.reset()
                print("Robot 1 left the ring!")
                return robot1_dot, robot2_dot
            if not robot2.is_in_ring():
                robot1.reset()
                robot2.reset()
                print("Robot 2 left the ring!")
                return robot1_dot, robot2_dot

            # Increment step; after every FRAMES_PER_ACTION steps, move to the next action
            step += 1
            if step % FRAMES_PER_ACTION == 0:
                action_index += 1  # Move to the next action

        return robot1_dot, robot2_dot  # Always return the artists

    def update_genetic_algorithm(frame):
        return update(frame)

    # Create the animation
    ani = FuncAnimation(fig, update_genetic_algorithm, init_func=init, blit=True, 
                        frames=FRAMES_PER_ACTION * NUM_ACTIONS * NUM_GENERATIONS, interval=20)
    plt.legend()
    plt.show()

plot_simulation()
