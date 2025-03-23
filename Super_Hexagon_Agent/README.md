<p align="center">
  <img src="assets/thumbnail.jpg" width="400x" />
</p>

# Super Hexagon Agent

An agent trained to play *Super Hexagon* by leveraging Policy Gradient methods for decision-making and continuous action optimization. This project integrates OpenGL hooks and memory injection to interface directly with the game and extract relevant states for reinforcement learning.

---

## Game Mechanics

### Environment
- **Game**: *Super Hexagon*, a fast-paced reflex-based game with a rotating hexagonal field.
- **State Variables**:
    - Player angle.
    - Obstacle, count, positions and angles.
    - World rotation.
    - Game State (Game Over, Level, Score, etc. ).
- **Action Space**:
    - Left
    - Right
    - Stay

### Integration Tools
- **OpenGL Hook**:  
  Utilizes the OpenGL hook from [polarbart/SuperHexagonAI](https://github.com/polarbart/SuperHexagonAI) to:
    - Start and stop the game.
    - Automate gameplay episodes for training purposes, like pause and fast-forward.

- **Memory Injection**:  
  Leverages [adrianchifor/super-hexagon-ai](https://github.com/adrianchifor/super-hexagon-ai) to:
    - Access the game's memory and extract specific state information, including player and obstacle positions.
    - Write values to memory (only for setup reasons, like level selection and player input).

## Algorithm: Proximal Policy Optimization (PPO)

## Documentation
The documentation for the implementation can be access on the following site: [https://stinktopf.github.io/SuperHexagonAI/](https://stinktopf.github.io/SuperHexagonAI/)
