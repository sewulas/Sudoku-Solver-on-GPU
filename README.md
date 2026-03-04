# Sudoku Solver on GPU

CUDA-based Sudoku solver that accelerates the search for solutions using GPU parallelism.
The solver combines **Breadth-First Search (BFS)** expansion on the GPU with a **Depth-First Search (DFS)** fallback for deeper branches.

The goal of the project was to explore **GPU computing, parallel algorithms, and CUDA kernel optimization**.

---

## Overview

This solver processes multiple Sudoku boards simultaneously on the GPU.
The algorithm first explores the search space using **parallel BFS expansion**, generating child boards according to the **Minimum Remaining Values (MRV)** heuristic.

If the search tree becomes too large to fit in GPU memory, the solver switches to **parallel DFS backtracking**, where each thread explores a board independently.

---

## Key Features

* Parallel exploration of Sudoku boards on the GPU
* MRV (Minimum Remaining Values) heuristic for efficient branching
* Hybrid **BFS + DFS** solving strategy
* Warp-level reduction in CUDA kernels
* Atomic synchronization to safely store solutions
* Efficient bitmask representation of Sudoku constraints

---

## Technologies

* **CUDA**
* **C++**
* **Thrust (CUDA parallel algorithms library)**

Concepts used:

* GPU parallel programming
* Breadth-First Search (BFS)
* Depth-First Search (DFS)
* Warp shuffle reduction
* Atomic operations
* Bitmask constraint propagation

---

## Algorithm

1. **Input loading**

   * Sudoku boards are read from a text file.
   * Each board is converted into a compact representation using bitmasks for rows, columns, and blocks.

2. **GPU BFS phase**

   * Each board is assigned to a CUDA block.
   * The **MRV heuristic** finds the cell with the smallest number of possible digits.
   * Boards are expanded in parallel to generate child states.

3. **Solution detection**

   * If a board is fully solved, the solution is written atomically to the solution buffer.

4. **DFS fallback**

   * If BFS expansion exceeds memory limits, the solver switches to **parallel DFS backtracking**.

---

## Data Representation

Each board is represented using:

* `cells[81]` — board values
* `rows[9]` — bitmask of digits used in each row
* `cols[9]` — bitmask of digits used in each column
* `blocks[9]` — bitmask of digits used in each 3×3 block

This representation allows **constant-time constraint checking** using bitwise operations.

---

## Example Usage

Program usage:

```
sudoku method count input_file output_file
```

Example:

```
sudoku gpu 10000 sudoku_input.txt sudoku_output.txt
```

Parameters:

* **method** – `cpu` or `gpu`
* **count** – number of boards to process
* **input_file** – file containing Sudoku boards
* **output_file** – file where solutions are written

---

## Input Format

Each line in the input file represents one Sudoku board:

```
530070000600195000098000060800060003400803001700020006060000280000419005000080079
```

* `0` represents an empty cell
* Each line contains exactly **81 digits**

---

## Output Format

The output file contains solved boards in the same format:

```
534678912672195348198342567859761423426853791713924856961537284287419635345286179
```

If a board cannot be solved, the output line contains zeros.

---

## Performance

The GPU solver can process **thousands of Sudoku boards simultaneously**, leveraging GPU parallelism to accelerate search.

Kernel timings are collected for:

* MRV computation
* board classification
* solution detection
* prefix scan
* board expansion
* memory transfers

---

## Author

Filip Sewastianowicz

Student of Computer Science
Warsaw University of Technology

---

## License

This project was created for educational purposes.
