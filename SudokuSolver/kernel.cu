
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

#define METHOD_CPU 0
#define METHOD_GPU 1

#define BOARD_DIM 9
#define BOARD_DIM2 81
#define BFS_DEPTH 3

#define FULL_CELL_VAL 100
#define DEAD_BOARD BOARD_DIM2
#define PARENT_SOLVED_TRUE 1
#define PARENT_SOLVED_FALSE 0

#define DEAD_BOARD_FLAG 0
#define ACTIVE_BOARD_FLAG 1
#define SOLVED_BOARD_FLAG 2

#define IDX_ERR 99

// struktura/funktor do exclusive scana
struct uint8_to_uint32
{
    __device__ uint32_t operator()(uint8_t x) const
    {
        return (uint32_t)x;
    }
};

// liczy i wypisuje statystyki czasowe kernela na podstawie
// próbek czasowych z arr[]
void printKernelStats(const char* name, float arr[], int N)
{
    float sum = 0.0f;
    float maxv = -1.0f;

    for (int i = 0; i < N; i++)
    {
        sum += arr[i];
        if (arr[i] > maxv)
            maxv = arr[i];
    }

    float avg = sum / N;

    printf("===== %s =====\n", name);
    printf("AVG: %.4f ms\n", avg);
    printf("MAX: %.4f ms\n", maxv);
}

// wypisuje cały raport dotyczący czasowje wydajności programu
void printTimePerformanceStats(float* time_FindMRV, float* time_markSolvedOrDead, float* time_checkForSolutions,
                            float* time_exclusiveScan, float* time_expandBoards, float* time_memcopy, float copying_time,
                            float writing_time, float elapsedTimeLoop, float time_dfs, bool wasDFSPerformed, int loops)
{
    printf("\nTime performance summary:\n");
    printf("===== Copying from input and preparing data format for further computations =====\n");
    printf("TOTAL: %.4f ms\n", copying_time);
    printf("===== Copying output to given output file =====\n");
    printf("TOTAL: %.4f ms\n", writing_time);
    printf("===== Total loop computations time =====\n");
    printf("TOTAL: %.4f ms\n", elapsedTimeLoop);
    printf("LOOPS: %d\n", loops);
    printf("===== Kernels' time performances (avg. and max) =====\n");
    printKernelStats("FindMRV", time_FindMRV, BOARD_DIM2);
    printKernelStats("markSolvedOrDead", time_markSolvedOrDead, BOARD_DIM2);
    printKernelStats("checkForSolutions", time_checkForSolutions, BOARD_DIM2);
    printKernelStats("exclusive_scan", time_exclusiveScan, BOARD_DIM2);
    printKernelStats("expandBoards", time_expandBoards, BOARD_DIM2);
    printKernelStats("memcopy", time_memcopy, BOARD_DIM2);
    if (wasDFSPerformed)
    {
        printf("===== solveBacktrack =====\nTOTAL: $.4f ms\n", time_dfs);
    }
    else
    {
        printf("===== solveBacktrack =====\nkernel didn't perform (compuations ended on BFS loop)\n");
        time_dfs = 0;
    }
    printf("===== TOTAL TIME (loop time + copying/io instructions) =====\n");
    float total = elapsedTimeLoop + copying_time + writing_time + time_dfs;
    printf("TOTAL: %.4f ms\n", total);
}

// funkcja służąca do debug'u, która wypisuje zawartość boardsCount plansz 
// będących zapisanych pod adresem d_cells na urządzeniu GPU
__host__ void debugPrintBoards(uint8_t* h_cells, uint8_t* d_cells, int boardsCount, uint16_t* parentID = NULL)
{
    cudaMemcpy(h_cells, d_cells, 81 * sizeof(uint8_t) * boardsCount, cudaMemcpyDeviceToHost);

    printf("\n===== DEBUG =====\n");
    int offset = BOARD_DIM2;
    for (int i = 0; i < boardsCount; i++)
    {
        printf("Board id = %d\n", i);
        if (parentID != NULL)
        {
            printf("Parent id = %d\n", parentID[i]);
        }
        for (int r = 0; r < 9; r++)
        {
            for (int c = 0; c < 9; c++)
            {
                printf("%d ", h_cells[r * 9 + c + i * offset]);
            }
            printf("\n");
        }
        printf("===========================================\n\n");
    }
}

__device__ inline int GetKthDigitFromMask(uint16_t mask, int k);

__device__ inline int popcount16(uint16_t x) { return __popc((unsigned)x); }

// funkcja pomocnicza konwertująca liczbę na maskę bitową wg konwencji
// 1 << digit (czyli zapalony bit numer digit = liczba digit znajduje
// znajduje się w tej masce)
__host__ __device__ uint16_t digit_to_mask(int digit)
{
    if (digit == 0)
        return 0x3FE; // 0b1111111110
    else
        return 1 << digit; //
}

// funkcja pomocnicza będąca odwrotną operacją do digit_to_mask(),
// z zastrzeżeniem, że jeżeli na masce jest więcej niż jedna cyfra
// zakodowana, to zwraca 0
__host__ __device__ int mask_to_digit(uint16_t cell)
{
#if defined(__CUDA_ARCH__)
    int msb = __clz(cell);
    int lsb = sizeof(unsigned int) * 8 - msb - 1;
#elif defined(_MSC_VER)
    unsigned long index;
    _BitScanForward(&index, cell);
    int lsb = (int)index;
#else
    int lsb = __builtin_ctz(cell);
#endif

    if (cell == 1 << lsb)
        return lsb;
    else
        return 0;
}

// funkcja obliczająca "Remaining Value" maski komórki, czyli
// ile 
__host__ __device__ int cellsRV(uint16_t mask)
{
    uint8_t count = 0;
    while (mask)
    {
        mask &= (mask - 1);
        count++;
    }
    return count;
}

// funkcja służąca do obliczenia indeksu bloku, na podstawie podanych indeksów
// wiersza (row) oraz kolumny (col)
__device__ __host__ int block_index(int row, int col) {
    return (row / 3) * 3 + (col / 3);
}

// funkcja wczytująca z pliku filename liczbę plansz (maxBoards) i interpretująca
// dane poprzez odpowiednie wypełnienie struktury cells oraz masek.
// UWAGA: funkcja zwraca liczbę poprawnie wczytanych plansz i informuje
// o liczbie niepoprawnych danych wejściowych
// Funkcja również wczytuje nie więcej niż maxBoards, to znaczy, że jeżeli
// maxBoards jest większe niż faktyczna liczba plansz to wczytuje tylko tyle, 
// ile plansz ma dany plik
__host__ int read_sudoku_file(const char* filename,
    int max_boards,
    uint8_t* cells_out,
    uint16_t* rows_out,
    uint16_t* cols_out,
    uint16_t* blocks_out)
{
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Nie udało się otworzyć pliku: %s\n", filename);
        return -1;
    }

    char line[128];
    int count = 0;
    int invalidCount = 0;
    int board_offset = 81;
    int mask_offset = 9;

    while (fgets(line, sizeof(line), f) && count < max_boards) {
        // pomiń puste linie
        if (strlen(line) < 81)
        {
            invalidCount++;
            continue;
        }

        for (int i = 0; i < BOARD_DIM * BOARD_DIM; i++) {
            char c = line[i];
            if (c < '0' || c > '9')
            {
                invalidCount++;
                break;
            }
            int digit = c - '0';
            cells_out[i + board_offset*count] = digit;

            if (digit != 0) {
                int row = i / 9;
                int col = i % 9;
                int blk = block_index(row, col);
                uint16_t bit = 1 << digit;
                rows_out[row + mask_offset * count] |= bit;
                cols_out[col + mask_offset * count] |= bit;
                blocks_out[blk + mask_offset * count] |= bit;
            }
        }
        count++;
    }
    fclose(f);
    printf("> Wczytano %d plansz Sudoku z pliku %s\n", count, filename);
    printf("> %d plansz mialo niepoprawne znaki\n", invalidCount);
    return count;
}

// funkcja przenosząca dane z Device na Hosta, a następnie zapisująca wynik
// do pliku filename według schematu: jeden wiersz = jedna plansza, zaś
// plansze są rozdzielone według konwencji przejścia do nowej linii z Windows (znaki CR+LF)
__host__ int write_sudoku_output(const char* filename,
    int max_boards,
    uint8_t* d_solved, uint8_t* d_isParentSolved)
{
    // Alokacja RAM
    uint8_t* h_solvedBoards = (uint8_t*)malloc(sizeof(uint8_t) * max_boards * BOARD_DIM2);
    cudaMemcpy(h_solvedBoards, d_solved,
        sizeof(uint8_t) * max_boards * BOARD_DIM2,
        cudaMemcpyDeviceToHost);

    uint8_t* h_isParentSolved = (uint8_t*)malloc(sizeof(uint8_t) * max_boards);
    cudaMemcpy(h_isParentSolved, d_isParentSolved,
        sizeof(uint8_t) * max_boards,
        cudaMemcpyDeviceToHost);

    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Nie udało się otworzyć pliku: %s\n", filename);
        return -1;
    }

    // Bufor na jedną linię (81 znaków + CRLF + terminator)
    char line[BOARD_DIM2 + 4];

    for (int board = 0; board < max_boards; board++)
    {
        // Jeśli plansza została rozwiązana:
        if (h_isParentSolved[board] == PARENT_SOLVED_TRUE)
        {
            for (int i = 0; i < BOARD_DIM2; i++)
            {
                uint8_t digit = h_solvedBoards[board * BOARD_DIM2 + i];
                line[i] = (digit >= 0 && digit <= 9) ? ('0' + digit) : '0';
            }
        }
        else
        {
            // Plansza nierozwiązana → wypisujemy same '0'
            memset(line, '0', BOARD_DIM2);
        }

        // CRLF zgodnie ze specyfikacją
        line[BOARD_DIM2] = '\r';
        line[BOARD_DIM2 + 1] = '\n';
        line[BOARD_DIM2 + 2] = '\0';

        fputs(line, f);
    }

    fclose(f);

    free(h_solvedBoards);
    free(h_isParentSolved);

    printf("> Has saved output to the file: %s\n", filename);
    return max_boards;
}

// Kernel liczący dla każdej z plansz w d_cells indeks komórki o MRV, czyli komórkę o "Minimum Remaining Values",
// co oznacza, że znajudje dla każdej planszy równolegle komórkę o najmniejszej liczbie możliwych cyfr do wpisania.
// Jeżeli znajdzie, to jej indeks jest wpisywany w odpowiednie miejsce w tablicy index_out.
// Jeżeli nie to wpisuje wartości: flagę DEAD_BOARD, kiedy albo parent tej planszy został rozwiązany albo
// znaleziono pustą komórkę o MRV = 0 => sprzeczność;
// Funkcja również zapisuje ile wynosi znalezione MRV do odpowiedniego miejsca w tablicy children_out.
// Jeżeli s_count == 0 => index_out = DEAD_BOARD.
// Jeżeli rodzic tej planszy został już rozwiązany, plansza jest oznaczana do odrzucenia (ustawiane są
// wartości children_out = 0 oraz index_out = DEAD_BOARD.
// Kernel wywoływany jest <<<liczba obecnych plansz, 96>>>, gdzie każdy wątek zajmuje się obliczeniem MRV dla
// swojej komórki (robi to dokładnie 81 wątków, ale wyrównane jest do 3 warpów).
// Następnie wykonywana jest redukcja na poziomie warpów, które komunikują się między sobą za pomocą
// __shfl_down_sync
__global__ void FindMRV_kernel(uint8_t* d_cells, uint16_t* d_rows, uint16_t* d_cols, uint16_t* d_blocks,
    uint8_t* index_out, uint8_t* children_out, uint32_t* d_parentID, uint8_t* d_isParentSolved)
{
    int boardID = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ uint8_t s_cells[81];
    __shared__ uint16_t s_rows[9];
    __shared__ uint16_t s_cols[9];
    __shared__ uint16_t s_blocks[9];

    __shared__ uint8_t s_count[81];
    __shared__ uint8_t s_idx[81];

    __shared__ uint8_t dead_flag;

    // tylko 81 pierwszyh wątków przepisuje dane do pamięci współdzielonej
    if (tid < BOARD_DIM2)
    {
        s_cells[tid] = d_cells[boardID * BOARD_DIM2 + tid];
        s_count[tid] = FULL_CELL_VAL; // defualt, for already filled cells
    }
    if (tid < BOARD_DIM)
    {
        s_rows[tid] = d_rows[boardID * BOARD_DIM + tid];
        s_cols[tid] = d_cols[boardID * BOARD_DIM + tid];
        s_blocks[tid] = d_blocks[boardID * BOARD_DIM + tid];
    }
    if (tid == 0)
    {
        // jeżeli rodzica planszy już rozwiązano, oznacz planszę jako DEAD_BOARD
        dead_flag = 0;
        if (d_isParentSolved[d_parentID[boardID]] == PARENT_SOLVED_TRUE)
        {
            index_out[boardID] = DEAD_BOARD;
            children_out[boardID] = 0;
            dead_flag = 1;
        }
    }
    __syncthreads();
    if (dead_flag != 0)
    {
        if (tid == 0)
        {
            index_out[boardID] = DEAD_BOARD;
            children_out[boardID] = 0;
        }
        return;
    }
    __syncthreads();
    // logika liczenia MRV każdej pustej komórki
    if (tid < 81)
    {
        if (s_cells[tid] == 0)
        {
            uint8_t col = tid % BOARD_DIM;
            uint8_t row = tid / BOARD_DIM;
            uint8_t block = block_index(row, col);
            uint16_t mask = s_rows[row] | s_cols[col] | s_blocks[block];
            s_count[tid] = popcount16((~mask) & 0x3FE);
            if (s_count[tid] == 0)
            {
                dead_flag = 1; // zaznacz jako DEAD_BOARD
            }
        }
        s_idx[tid] = tid;
    }
    __syncthreads();
    if (dead_flag != 0)
    {
        if (tid == 0)
        {
            index_out[boardID] = DEAD_BOARD;
            children_out[boardID] = 0;
        }
        return;
    }

    // sprawdz czy tid < 81
    bool inRange = (tid < BOARD_DIM2);

    // wartości MRV i indeksy zapisane do pamięci lokalnej wątków, aby następnie
    // komunikować się między warpami
    int val32 = inRange ? (int)s_count[tid] : (int)FULL_CELL_VAL;
    int idx32 = inRange ? (int)s_idx[tid] : (int)BOARD_DIM2;
    unsigned fullMask = 0xFFFFFFFF;

    // warp shuffle reduce, używając maski fullMask, aby wyłonić najlepsze wartości dla bloków
    // indeksów 0...31, 32...63, 64...80
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_val = __shfl_down_sync(fullMask, val32, offset);
        int other_idx = __shfl_down_sync(fullMask, idx32, offset);
        if (other_val < val32 || (other_val == val32 && other_idx < idx32)) {
            val32 = other_val;
            idx32 = other_idx;
        }
    }

    // zapisujemy najlepsze wyniki do początków bloków obliczeń, czyli 0, 32 i 64
    if ((tid % 32) == 0) 
    {
        s_count[tid] = val32;
        s_idx[tid] = idx32;
    }
    __syncthreads();

    // finalna redukcja wyników warpów (maks 3) przez thread 0
    if (tid == 0) {
        int best_val = (int)s_count[0];
        int best_idx = (int)s_idx[0];
        for (int i = 1; i < 3; ++i) {
            int cv = (int)s_count[i*32];
            int ci = (int)s_idx[i*32];
            if (cv < best_val || (cv == best_val && ci < best_idx)) {
                best_val = cv;
                best_idx = ci;
            }
        }
        if (best_val == FULL_CELL_VAL) {
            index_out[boardID] = (uint8_t)FULL_CELL_VAL;
            children_out[boardID] = 0;
        }
        else {
            index_out[boardID] = (uint8_t)best_idx;
            children_out[boardID] = (uint8_t)best_val;
        }
    }
    return;
}

// Funkcja pomocnicza do inicjalizacji wskazanych przez parametry tablic
__global__ void initAuxilaryArrays(uint8_t* d_solved, uint32_t* parentID, uint8_t* d_isParentSolved, int* d_parent_lock, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count)
    {
        parentID[tid] = tid;
        d_isParentSolved[tid] = PARENT_SOLVED_FALSE;
    }
}

// Kernel oznaczający na podstawie wyników kernela FindMRV_kernel flagami:
// DEAD_BOARD_FLAG, SOLVED_BOARD_FLAG oraz ACTIVE_BOARD_FLAG odpowiednio, gdy:
// - plansza ma MRVindex == ACTIVE_BOARD_FLAG
// - plansza ma MRVindex == FULL_CELL_VAL
// - plansza ma count > 0
__global__ void markSolvedOrDead(uint8_t* d_MRV, uint8_t* d_childCount, uint8_t* d_flagActive,
    int maxCount, uint32_t* d_parentID)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= maxCount) return;
    uint8_t count = d_childCount[tid];
    uint8_t index = d_MRV[tid];

    if (count == 0)
    {
        if (index == DEAD_BOARD) 
        {
            d_flagActive[tid] = DEAD_BOARD_FLAG;
        }
        else if (index == FULL_CELL_VAL)
        {
            d_flagActive[tid] = SOLVED_BOARD_FLAG;
        }
        return;
    }
    else
    {
        d_flagActive[tid] = ACTIVE_BOARD_FLAG;
    }
}

// Kernel sprawdzający czy jakaś plansza została oznaczona przez kernel markSolvedOrDead. Jeżeli tak
// to atomowo zapisuje wynik do odpowiedniego miesjca w buforze na rozwiązania (d_solved). WYkoryztsuje
// przy tym atomicCAS oraz tablicę pomocniczych locków z d_parent_lock.
__global__ void checkForSolutions(uint8_t* d_cells, uint8_t* d_flagActive, uint8_t* d_solved, uint32_t* d_parentID,
                                int countMax, uint32_t* solvedCount, uint8_t* d_isParentSolved, int* d_parent_lock)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= countMax)
        return;
    int boardID = tid * BOARD_DIM2;
    uint8_t flag = d_flagActive[tid];
    uint32_t parentID = d_parentID[tid];
    if (flag == SOLVED_BOARD_FLAG)
    {
        int old = atomicCAS(&d_parent_lock[parentID], 0, 1);
        if (old == 0)
        {
            for (size_t i = 0; i < BOARD_DIM2; i++)
            {
                d_solved[parentID * BOARD_DIM2 + i] = d_cells[boardID + i];
            }
            atomicAdd(solvedCount, 1);
            d_isParentSolved[parentID] = PARENT_SOLVED_TRUE;
        }
    }
}

// Kernel odpowiadający za prawidłowe przepisanie dzieci plansz z buforów d_cells itp do buforów
// *_children. Wykorzystuje przy tym kluczowe informacje z tablic d_offset oraz d_parentID.
// Jedna plansza z bufora obliczeniowego jest obsługiwana przez jeden wątek, który tworzy tyle dzieci,
// ile wskazuje na to tablica d_childrenCount z kernela FindMRV_kernel.
__global__ void expandBoards(uint8_t* d_cells, uint16_t* d_rows, uint16_t* d_cols, uint16_t* d_blocks,
    uint8_t* d_cells_out, uint16_t* d_rows_out, uint16_t* d_cols_out, uint16_t* d_blocks_out,
    uint8_t* d_MRV, uint8_t* d_childrenCount, uint32_t* d_offset, uint8_t* d_flagActive, uint8_t* d_solved,
    uint32_t* d_parentID, int countMax, uint32_t* solvedCount, uint8_t* d_isParentSolved, uint32_t* d_parentID_out,
    int levelDEBUG = 0)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= countMax)
        return;
    int boardID = tid * BOARD_DIM2;
    int maskID = tid * BOARD_DIM;
    uint8_t flag = d_flagActive[tid];
    uint32_t parentID = d_parentID[tid];
    uint32_t childOffset = d_offset[tid];
    uint8_t count = d_childrenCount[tid];

    uint8_t index = d_MRV[tid];
    uint16_t* rowMask, *colMask, *blockMask;
    rowMask = (d_rows + maskID);
    colMask = (d_cols + maskID);
    blockMask = (d_blocks + maskID);

    if (flag == DEAD_BOARD_FLAG)
    {
        return; // do nothing
    }
    else if (flag == SOLVED_BOARD_FLAG)
    {
        return;
    }

    uint8_t row, col, block;
    col = index % BOARD_DIM;
    row = index / BOARD_DIM;
    block = block_index(row, col);
    uint16_t mask = rowMask[row] | colMask[col] | blockMask[block];
    uint16_t avaiable = (~mask) & 0x3FE;
    for (size_t k = 0; k < count; k++)
    {
        int childIdx = childOffset * BOARD_DIM2 + k * BOARD_DIM2;
        int childMaskIdx = childOffset * BOARD_DIM + k * BOARD_DIM;
        for (size_t i = 0; i < BOARD_DIM2; i++)
        {
            d_cells_out[childIdx + i] = d_cells[boardID + i];
            if (i % BOARD_DIM == 0)
            {
                d_rows_out[childMaskIdx + i / BOARD_DIM] = d_rows[maskID + i / BOARD_DIM];
                d_cols_out[childMaskIdx + i / BOARD_DIM] = d_cols[maskID + i / BOARD_DIM];
                d_blocks_out[childMaskIdx + i / BOARD_DIM] = d_blocks[maskID + i / BOARD_DIM];
            }
        }
        uint8_t digit = GetKthDigitFromMask(avaiable, k);
        d_cells_out[childIdx + index] = digit;
        d_rows_out[childMaskIdx + row] |= digit_to_mask(digit);
        d_cols_out[childMaskIdx + col] |= digit_to_mask(digit);
        d_blocks_out[childMaskIdx + block] |= digit_to_mask(digit);
    }
    __syncthreads();
    // distribuite respective parentID to your children:
    for (size_t i = 0; i < count; i++)
    {
        d_parentID_out[childOffset + i] = parentID;
    }
}

// Iteracyjna wersja FindMRV_kernel wykorzystana w kernelu solveBacktrack.
__device__ uint8_t FindMRV(uint8_t* d_cells, uint16_t* d_rows, uint16_t* d_cols, uint16_t* d_blocks)
{
    uint8_t idx = FULL_CELL_VAL;
    uint8_t minVal = FULL_CELL_VAL;
    uint8_t val, col, row, block;
    uint16_t mask, available;
    for (int i = 0; i < BOARD_DIM2; i++)
    {
        if (d_cells[i] != 0)
            continue;
        col = i % BOARD_DIM;
        row = i / BOARD_DIM;
        block = block_index(row, col);
        mask = d_rows[row] | d_cols[col] | d_blocks[block];
        available = (~mask) & 0x3FE;
        val = popcount16(available);

        if (val < minVal || (val == minVal && i < idx))
        {
            minVal = val;
            idx = i;
        }
    }

    if (minVal <= 0)
        idx = IDX_ERR; // wykryto sprzeczność! = pole bez żadnej możliwej cyfry do wpisania

    return idx;
}

// Funkcja pobiera k-tą liczbę, która jest obecna w masce (idąc od najmniejszej liczby)
__device__ inline int GetKthDigitFromMask(uint16_t mask, int k)
{
    for (int i = 1; i <= BOARD_DIM; i++)
    {
        uint16_t temp = digit_to_mask(i);
        if ((temp & mask) != 0)
        {
            if (k == 0)
                return i;
            else
                k--;
        }
    }
    return -1; // error
}

// Kernel, który przydziela jednemu wątku jedną planszę do backtrackingowego przeszukania drzewa rozwiązań planszy
// w głąb (DFS). Wykorzystuje funkcję FindMRV do wyznaczenia pola o MRV, a następnie schodzi w dół z nawrotami, 
// przy czym wykorzystuje zaalokowane stosy.
__global__ void solveBacktrack(uint8_t* d_cells, uint16_t* d_rows, uint16_t* d_cols, uint16_t* d_blocks,
                            uint32_t* d_parentID, uint8_t* d_solved, uint8_t* d_isParentSolved, int* d_parent_lock,
                            uint32_t* solvedCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cells_offset = idx * BOARD_DIM2;
    int mask_offset = idx * BOARD_DIM;
    uint32_t parentID = d_parentID[idx];

    __syncthreads(); // check if necessary
    uint8_t col, row, block;
    uint16_t mask, available;

    // tutaj rozmiar tablic możnaby zmniejszyć w zależności tego ile BFS zrobiliśmy
    int DFSdepth = 0;
    uint8_t digits_stack[BOARD_DIM2];
    uint8_t index_stack[BOARD_DIM2];
    uint8_t left_stack[BOARD_DIM2];

    uint8_t index = 0;
    index = FindMRV(d_cells + cells_offset, d_rows + mask_offset, d_cols + mask_offset, d_blocks + mask_offset);
    index_stack[0] = index;

    col = index % BOARD_DIM;
    row = index / BOARD_DIM;
    block = block_index(row, col);
    mask = d_rows[mask_offset + row] | d_cols[mask_offset + col] | d_blocks[mask_offset + block];
    available = (~mask) & 0x3FE;
    uint8_t left = popcount16(available) - 1;
    uint8_t digit = GetKthDigitFromMask(available, 0);
    left_stack[0] = left;
    digits_stack[0] = digit;

    d_cells[cells_offset + index] = digit;
    d_rows[mask_offset + row] |= digit_to_mask(digit);
    d_cols[mask_offset + col] |= digit_to_mask(digit);
    d_blocks[mask_offset + block] |= digit_to_mask(digit);

    while (DFSdepth >= 0 && (d_isParentSolved[parentID] == PARENT_SOLVED_FALSE))
    {
        index = FindMRV(d_cells + cells_offset, d_rows + mask_offset, d_cols + mask_offset, d_blocks + mask_offset);
        if (index == FULL_CELL_VAL) // solved
        {
            printf("Znaleziono rozwiazanie!\n");
            int old = atomicCAS(&d_parent_lock[parentID], 0, 1);
            if (old == 0)
            {
                for (size_t i = 0; i < BOARD_DIM2; i++)
                {
                    d_solved[parentID * BOARD_DIM2 + i] = d_cells[cells_offset + i];
                }
                atomicAdd(solvedCount, 1);
                d_isParentSolved[parentID] = PARENT_SOLVED_TRUE;
            }
            break;
        }
        else if (index == IDX_ERR)
        {
            // reverse it
            do
            {
                if (DFSdepth < 0)
                {
                    printf("[DEBUG] ABORTING DFS instnace...\n");
                    return; // = plansza bez rozwiązania
                }

                // pop ze stacka
                index = index_stack[DFSdepth];
                left = left_stack[DFSdepth];
                digit = digits_stack[DFSdepth];

                col = index % BOARD_DIM;
                row = index / BOARD_DIM;
                block = block_index(row, col);
                mask = digit_to_mask(digit);

                // wracanie do stanu przed
                d_cells[cells_offset + index] = 0;
                d_rows[mask_offset + row] &= ((~mask) & 0x3FE);
                d_cols[mask_offset + col] &= ((~mask) & 0x3FE);
                d_blocks[mask_offset + block] &= ((~mask) & 0x3FE);
                
                DFSdepth--;
            } while (left < 1);
            DFSdepth++;
            col = index % BOARD_DIM;
            row = index / BOARD_DIM;
            block = block_index(row, col);

            // jesteśmy w cell[index] z left >= 1, więc mamy przynajmniej jeszcze jedną opcję
            mask = d_rows[mask_offset + row] | d_cols[mask_offset + col] | d_blocks[mask_offset + block];
            available = (~mask) & 0x3FE;
            uint8_t total = popcount16(available);
            digit = GetKthDigitFromMask(available, total - left); // total - left git?
            left--;

            index_stack[DFSdepth] = index;
            left_stack[DFSdepth] = left;
            digits_stack[DFSdepth] = digit;
            mask = digit_to_mask(digit);
            d_cells[cells_offset + index] = digit;
            d_rows[mask_offset + row] |= mask;
            d_cols[mask_offset + col] |= mask;
            d_blocks[mask_offset + block] |= mask;
        }
        else
        {
            DFSdepth++;

            index_stack[DFSdepth] = index;
            col = index % BOARD_DIM;
            row = index / BOARD_DIM;
            block = block_index(row, col);
            mask = d_rows[mask_offset + row] | d_cols[mask_offset + col] | d_blocks[mask_offset + block];
            available = (~mask) & 0x3FE;
            left = popcount16(available) - 1;
            digit = GetKthDigitFromMask(available, 0);
            left_stack[DFSdepth] = left;
            digits_stack[DFSdepth] = digit;
            mask = digit_to_mask(digit);
            d_cells[cells_offset + index] = digit;
            d_rows[mask_offset + row] |= mask;
            d_cols[mask_offset + col] |= mask;
            d_blocks[mask_offset + block] |= mask;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        printf("Incorrect program execution.\r\n");
        printf("USAGE:\r\n");
        printf("    sudoku method count input_file output_file\r\n\r\n");
        printf("where:\r\n");
        printf("  method      = cpu or gpu\r\n");
        printf("  count       = amount of boards to solve (positive integer)\r\n");
        printf("  input_file  = input file for sudoku boards\r\n");
        printf("  output_file = output file\r\n");
        return 1;
    }
    const char* method_str = argv[1];
    const char* count_str = argv[2];
    const char* input_file = argv[3];
    const char* output_file = argv[4];
    
    int method = -1;
    if (strcmp(method_str, "cpu") == 0)
        method = METHOD_CPU;
    else if (strcmp(method_str, "gpu") == 0)
        method = METHOD_GPU;
    else
    {
        printf("Incorrect parameter 'method'. Try: cpu or gpu\r\n");
        return 1;
    }
    int read_count = atoi(count_str);
    if (read_count <= 0)
    {
        printf("Parameter 'count' must be a postitive integer.\r\n");
        return 1;
    }

    printf("Executed progarm with:\r\n");
    printf("  method      = %s\r\n", method_str);
    printf("  count       = %d\r\n", read_count);
    printf("  input_file  = %s\r\n", input_file);
    printf("  output_file = %s\r\n", output_file);

    if (method == METHOD_CPU)
    {
        printf("[INFO] Chosen method: CPU.\r\n");
        printf("       Not implemented (yet).\r\n");
        // TODO: CPU
        // read_sudoku_input() → solve() → write_sudoku_output()
        return 0;
    }
    printf("[INFO] Chosen method GPU.\r\n");
    printf("       Proceeding to start computation.\r\n");

    // ============= GPU Version =============
    // statyczne tablice do pomiaru czasu
    static float time_FindMRV[BOARD_DIM2] = { 0 };
    static float time_markSolvedOrDead[BOARD_DIM2] = { 0 };
    static float time_checkForSolutions[BOARD_DIM2] = { 0 };
    static float time_exclusiveScan[BOARD_DIM2] = { 0 };
    static float time_expandBoards[BOARD_DIM2] = { 0 };
    static float time_memcopy[BOARD_DIM2] = { 0 };

    // debug mode dla większej liczby komunikatów o działaniu programu
    bool debugModeOn = false;

    ////// hardcode'owane dane wejściowe:
    //int read_count = 333590;
    //const char* input_file = "sudoku_data.txt"; // sudoku_data.csv
    //const char* output_file = "sudoku_output.txt";

    // obliczanie dostępnej pamięci na urządzeniu
    size_t boardSize = sizeof(uint8_t) * BOARD_DIM2 + 3 * BOARD_DIM * sizeof(uint16_t);
    if(debugModeOn)
        printf("[DEBUG] Rozmiar jednej plansz w pamieci: %dB\n", boardSize);

    float free_m, total_m, used_m;
    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);

    float free_b = (uint32_t)free_t;
    float allocated = free_b * 0.8;
    if (debugModeOn)
        printf("[DEBUG] Free memory: %fB\nUsed for allocation: %fB\n", free_b, allocated);
    free_m = (uint32_t)free_t / 1048576.0;
    total_m = (uint32_t)total_t / 1048576.0;
    used_m = total_m - free_m;
    if (debugModeOn)
        printf("[DEBUG] mem free %f MB\nmem total %f MB\nmem used %f MB\n", free_m, total_m, used_m);
    int possibleCount = free_b / boardSize;
    int allocatedCount = allocated / boardSize;
    if (debugModeOn)
        printf("[DEBUG] Possible count of boards: %d\nTo be allocated count of boards: %d\n", possibleCount, allocatedCount);
    
    // alokacja miejsca na plansze na hoście
    uint8_t* h_cells = (uint8_t*)malloc(81 * sizeof(uint8_t) * read_count);
    uint16_t* h_rows = (uint16_t*)malloc(9 * sizeof(uint16_t) * read_count);
    uint16_t* h_cols = (uint16_t*)malloc(9 * sizeof(uint16_t) * read_count);
    uint16_t* h_blocks = (uint16_t*)malloc(9 * sizeof(uint16_t) * read_count);

    memset(h_rows, 0, sizeof(uint16_t) * 9 * read_count);
    memset(h_cols, 0, sizeof(uint16_t) * 9 * read_count);
    memset(h_blocks, 0, sizeof(uint16_t) * 9 * read_count);

    // kopiowanie z pliku do CPU:
    int current_count = read_sudoku_file(input_file, read_count, h_cells, h_rows, h_cols, h_blocks); // liczba poprawnie wczytanych plansz
    read_count = current_count;
    float memoryForBuffers = allocated - current_count * BOARD_DIM2 * sizeof(uint8_t);
    int maxBoards = memoryForBuffers/boardSize/2;
    if (debugModeOn)
        printf("[DEBUG] Max boards used to compute with BFS: %d (%f MB)\nWhich is 2 buffers each with %d boards\n", maxBoards * 2, memoryForBuffers / 1048576.0, maxBoards);

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // bufor na rozwiązane plansze
    uint8_t* d_solved;
    cudaError_t err = cudaMalloc((void**)&d_solved, sizeof(uint8_t) * BOARD_DIM2 * current_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemset(d_solved, 0, sizeof(uint8_t) * current_count * BOARD_DIM2);

    // zmienna do przechowywania liczby aktualnie rozwiązanych plansz
    uint32_t* d_solvedCount;
    err = cudaMalloc((void**)&d_solvedCount, sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // zmienne do atomicCAS, do gwarancji atomowości zapisu rozwiązania w d_solved
    int* d_parent_lock;
    err = cudaMalloc((void**)&d_parent_lock, sizeof(int) * current_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemset(d_parent_lock, 0, sizeof(int) * current_count);

    // tablica przechowująca informację od jakiej planszy z pliku pochodzi dana plansza
    // w drzewie poszukiwań BFS
    // ZMIENIAMY PARENTID NA UINT32_t zeby pomiescic wiecej
    // elo
    uint32_t* d_parentID;
    err = cudaMalloc((void**)&d_parentID, sizeof(uint32_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // tablica - bufor dla nowego pokolenia po expandBoards
    uint32_t* d_parentID_children;
    err = cudaMalloc((void**)&d_parentID_children, sizeof(uint32_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // tablica informująca, która plansza z pliku input została już rozwiązana
    uint8_t* d_isParentSolved;
    err = cudaMalloc((void**)&d_isParentSolved, sizeof(uint8_t) * current_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // inicjacja tablic pomocniczych (d_solved, d_parentID itd.)
    int blockSize1 = 256;                     
    int gridSize1 = (current_count + blockSize1 - 1) / blockSize1;
    initAuxilaryArrays << <gridSize1, blockSize1 >> > (d_solved, d_parentID,d_isParentSolved, d_parent_lock, current_count); // inicjacja komórek ze śmieci na '0'
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(e)); }
    cudaDeviceSynchronize();

    // do kernela MRV
    uint8_t* d_MRVindecies;     // indeks pola o MRV dla i-tej planszy
    uint8_t* d_childrenCount;   // liczba dzieci z danej i-tej planszy
    err = cudaMalloc((void**)&d_MRVindecies, sizeof(uint8_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void**)&d_childrenCount, sizeof(uint8_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // do klasyfikacji
    uint8_t* d_activeFlag;  // tablica przechwoująca stan planszy: - DEAD_BOARD_FLAG 0   = plansza sprzeczna
                                                                // - ACTIVE_BOARD_FLAG 1 = plansza gotowa do dalszych przeszukiwań
                                                                // - SOLVED_BOARD_FLAG 2 = plansza oznaczona do przeniesienia jako rozwiązanie
    err = cudaMalloc((void**)&d_activeFlag, sizeof(uint8_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Bardzo ważna tablica, która będzie przechpwywać wynik pre scana na
    // tablicy d_chilrenCount. Mówi ona z jakim offsetem należy zalokować planszę
    // z jednego buforu do bufora z dziećmi
    uint32_t* d_offset;
    err = cudaMalloc((void**)&d_offset, sizeof(uint32_t) * maxBoards);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // kopiowanie do GPU
    // reprezentacja planszy (bufor do obliczeń)
    uint8_t* d_cells = NULL;
    uint16_t* d_rows = NULL;
    uint16_t* d_cols = NULL;
    uint16_t* d_blocks = NULL;

    // bufor do zapisania dzieci
    uint8_t* d_cells_children = NULL;
    uint16_t* d_rows_children = NULL;
    uint16_t* d_cols_children = NULL;
    uint16_t* d_blocks_children = NULL;

    // ============= alokacja pamięci na GPU =============
    float time_copying;
    cudaEvent_t startCopying, stopCopying;
    cudaEventCreate(&startCopying);
    cudaEventCreate(&stopCopying);
    cudaEventRecord(startCopying, 0);

    size_t sizeForCells = maxBoards * sizeof(uint8_t) * BOARD_DIM2;
    size_t current_size = current_count * sizeof(uint8_t) * BOARD_DIM2;

    // alokacja komórek
    err = cudaMalloc((void**)&d_cells, sizeForCells);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_cells_children, sizeForCells);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_cells, h_cells, current_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error : % s\n", cudaGetErrorString(err));
        cudaFree(d_cells);
        return 1;
    }

    // alokacja masek:
    size_t sizeForMasks = maxBoards * sizeof(uint16_t) * BOARD_DIM;
    current_size = current_count * sizeof(uint16_t) * BOARD_DIM;
    // rows:
    err = cudaMalloc((void**)&d_rows, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_rows_children, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_rows, h_rows, current_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error : % s\n", cudaGetErrorString(err));
        cudaFree(d_rows);
        return 1;
    }

    // cols:
    err = cudaMalloc((void**)&d_cols, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_cols_children, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_cols, h_cols, current_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error : % s\n", cudaGetErrorString(err));
        cudaFree(d_cols);
        return 1;
    }

    // blocks:
    err = cudaMalloc((void**)&d_blocks, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_blocks_children, sizeForMasks);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_blocks, h_blocks, current_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error : % s\n", cudaGetErrorString(err));
        cudaFree(d_blocks);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventRecord(stopCopying, 0);
    cudaEventSynchronize(stopCopying);
    cudaEventElapsedTime(&time_copying, startCopying, stopCopying);
    cudaEventDestroy(startCopying);
    cudaEventDestroy(stopCopying);
    int bytes = current_count * sizeof(uint8_t) * BOARD_DIM2 + 3 * current_count * sizeof(uint16_t) * BOARD_DIM;
    printf("> Copied %d Sudoku boards to the GPU (%.2f KB)\n", current_count, bytes / 1024.0);

    // ============= główne obliczenia na GPU =============
    int level = 0;  // liczba zejść BFS-a
    int blockSize = 256;
    int gridSize = (current_count + blockSize - 1) / blockSize;
    bool isDFSneeded = false; // flaga, mówiąca czy potrzebne jest wykonanie iteracyjnego
                              // backtrackingu (fallback)

    // pomiar czasu głównej pętli obliczeń
    float elapsedTimeMainLoop;
    cudaEvent_t startLoop, stopLoop;
    cudaEventCreate(&startLoop);
    cudaEventCreate(&stopLoop);
    cudaEventRecord(startLoop, 0);

    // Główna pętla obliczeń
    while (true)
    {
        // ============= ETAP 1 — FindMRV =============
        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            FindMRV_kernel << <current_count, 96 >> > (
                d_cells, d_rows, d_cols, d_blocks,
                d_MRVindecies, d_childrenCount, d_parentID, d_isParentSolved
                );

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);

            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_FindMRV[level] = ms;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        // ============= ETAP 2 - klasyfikacja =============
        blockSize = 256;
        gridSize = (current_count + blockSize - 1) / blockSize;
        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            markSolvedOrDead << <gridSize, blockSize >> > (d_MRVindecies, d_childrenCount, d_activeFlag, current_count, d_parentID);

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_markSolvedOrDead[level] = ms;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        blockSize = 256;
        gridSize = (current_count + blockSize - 1) / blockSize;
        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            checkForSolutions << <gridSize, blockSize >> > (d_cells, d_activeFlag, d_solved, d_parentID,
                current_count, d_solvedCount, d_isParentSolved, d_parent_lock);

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_checkForSolutions[level] = ms;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s at level=%d (after checkForSolutions_kernel)\n", cudaGetErrorString(err), level);
            return 1;
        }

        // ============= ETAP 3 - pre scan =============
        // castowanie z uint8 na uint32, żeby exclusive_scan poprawnie zadziałał
        auto first = thrust::make_transform_iterator(
            d_childrenCount,
            uint8_to_uint32()
        );

        auto last = first + current_count;

        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            thrust::exclusive_scan(
                thrust::device,
                first, last, d_offset
            );

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_exclusiveScan[level] = ms;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        // zmienna przechowująca informację o ilości dzieci w kolejnej iteracji
        // (po wykluczeniu plansz z flagą inna niż aktywna)
        int countNextGen = 0;

        // przeniesienie wyników na CPU w celu obliczenia countNextGen
        uint32_t* h_offset = (uint32_t*)malloc(sizeof(uint32_t) * current_count);
        uint8_t* h_childCount = (uint8_t*)malloc(sizeof(uint8_t) * current_count);
        cudaMemcpy(h_offset, d_offset, sizeof(uint32_t) * current_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_childCount, d_childrenCount, sizeof(uint8_t) * current_count, cudaMemcpyDeviceToHost);

        // ============= ETAP 4 - expandBoards =============

        countNextGen = h_offset[current_count - 1] + h_childCount[current_count - 1];
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        if (debugModeOn)
            printf("[DEBUG][LEVEL %d] countNextGen = %d\n", level, countNextGen);

        // jeżeli liczba dzieci w kolejnym pokoleniu przekracza maksymalną zaalokowaną liczbę
        // plansz w buforze, wyjdź z pętli i przejdź do Etapu 6 (fallback)
        if (countNextGen > maxBoards)
        {
            if (debugModeOn)
                printf("[DEBUG] countNextGen exceeds maxBoards count! Exiting BFS loop...\n");
            isDFSneeded = true; // przechodzimy do iteracyjnego backtrackingu
            break;
        }
        // jeżeli wszystkie plansze zostały oznaczone jako dead/solved
        // i nie ma już plansz do przetworzenia, uznaj obliczenia za skończone
        if (countNextGen < 1)
        {
            if (debugModeOn)
                printf("[DEBUG] No boards left to compute (ending BFS phase succesfully)\n");
            break;
        }

        blockSize = 256;
        gridSize = (current_count + blockSize - 1) / blockSize;
        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            expandBoards << <gridSize, blockSize >> > (
                d_cells, d_rows, d_cols, d_blocks,
                d_cells_children, d_rows_children, d_cols_children, d_blocks_children,
                d_MRVindecies, d_childrenCount, d_offset, d_activeFlag, d_solved, d_parentID, current_count,
                d_solvedCount, d_isParentSolved, d_parentID_children, level
                );

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_expandBoards[level] = ms;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize error: %s at level=%d (after expandBoards_kernel)\n", cudaGetErrorString(err), level);
            return 1;
        }

        if (countNextGen < 1)
        {
            if (debugModeOn)
                printf("[DEBUG] No boards left to compute (ending BFS phase succesfully)\n");
            break;
        }

        // ============= ETAP 5 - kopiowanie z bufora =============
        {
            cudaEvent_t evStart, evStop;
            cudaEventCreate(&evStart);
            cudaEventCreate(&evStop);

            cudaEventRecord(evStart);

            // przeniesienie danych z jednego bufora tymczasowego (*_children) do
            // bufora wykorzystywanego do obliczeń w pętli
            size_t newSizeCells = countNextGen * BOARD_DIM2 * sizeof(uint8_t);
            size_t newSizeMasks = countNextGen * BOARD_DIM * sizeof(uint16_t);
            size_t newSizeParentID = countNextGen * sizeof(uint32_t);
            cudaMemcpy(d_cells, d_cells_children, newSizeCells, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_rows, d_rows_children, newSizeMasks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_cols, d_cols_children, newSizeMasks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_blocks, d_blocks_children, newSizeMasks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_parentID, d_parentID_children, newSizeParentID, cudaMemcpyDeviceToDevice);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize error: %s\n", cudaGetErrorString(err));
                return 1;
            }

            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evStop);
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);

            time_memcopy[level] = ms;
        }

        // iteracja przez pętle
        current_count = countNextGen;
        level++;
    }
    cudaEventRecord(stopLoop, 0);
    cudaEventSynchronize(stopLoop);
    cudaEventElapsedTime(&elapsedTimeMainLoop, startLoop, stopLoop);
    cudaEventDestroy(startLoop);
    cudaEventDestroy(stopLoop);

    // ============= ETAP 6 (fallback) - iteracyjny backtarcking =============
    float time_DFS = 0;
    if (isDFSneeded)
    {
        cudaEventCreate(&startCopying);
        cudaEventCreate(&stopCopying);
        cudaEventRecord(startCopying, 0);

        blockSize = 256;                     
        gridSize = (current_count + blockSize - 1) / blockSize;

        solveBacktrack << <gridSize, blockSize >> > (
            d_cells,
            d_rows,
            d_cols,
            d_blocks,
            d_parentID, d_solved, d_isParentSolved, d_parent_lock, d_solvedCount);

        cudaEventRecord(stopCopying, 0);
        cudaEventSynchronize(stopCopying);
        cudaEventElapsedTime(&time_DFS, startCopying, stopCopying);
        cudaEventDestroy(startCopying);
        cudaEventDestroy(stopCopying);

        cudaDeviceSynchronize();
    }

    uint32_t h_solvedCount = 0;
    cudaMemcpy(&h_solvedCount, d_solvedCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("> Solved %d out of %d total boards\n", h_solvedCount, read_count);
    if (debugModeOn && h_solvedCount < read_count)
    {
        printf("Unsolved parentIDs:\n");
        uint8_t* h_isSolved = (uint8_t*)malloc(sizeof(uint8_t) * maxBoards);
        cudaMemcpy(h_isSolved, d_isParentSolved, sizeof(uint8_t)* read_count, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < read_count; i++)
        {
            if (h_isSolved[i] == PARENT_SOLVED_FALSE) {
                printf(" id=%d,", i);
            }
        }
    }

    // ============= Kopiowanie wyniku =============
    float time_write_output = 0;
    {
        cudaEvent_t evStart, evStop;
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);

        cudaEventRecord(evStart);

        write_sudoku_output(output_file, read_count, d_solved, d_isParentSolved);

        cudaEventRecord(evStop);
        cudaEventSynchronize(evStop);

        cudaEventElapsedTime(&time_write_output, evStart, evStop);
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
    }

    printTimePerformanceStats(time_FindMRV, time_markSolvedOrDead, time_checkForSolutions,
        time_exclusiveScan, time_expandBoards, time_memcopy, time_copying, time_write_output, 
        elapsedTimeMainLoop, time_DFS, isDFSneeded, level);

    // Zwolnienie zasobów
    cudaFree(d_blocks_children);
    cudaFree(d_cols_children);
    cudaFree(d_rows_children);
    cudaFree(d_cells_children);
    cudaFree(d_parentID_children);

    cudaFree(d_blocks);
    cudaFree(d_cols);
    cudaFree(d_rows);
    cudaFree(d_cells);

    cudaFree(d_activeFlag);
    cudaFree(d_offset);
    cudaFree(d_childrenCount);
    cudaFree(d_MRVindecies);

    cudaFree(d_parent_lock);
    cudaFree(d_isParentSolved);
    cudaFree(d_parentID);
    cudaFree(d_solvedCount);
    cudaFree(d_solved);

    free(h_blocks);
    free(h_cols);
    free(h_rows);
    free(h_cells);
    return 0;
}