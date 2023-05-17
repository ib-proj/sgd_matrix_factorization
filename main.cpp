#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <chrono>
#include <random>
#include <fstream>
#include <atomic>

using namespace std;

vector<vector<double>> block_diagonal_matrix(int n, int block_size, double min_val, double max_val) {
    if (n % block_size != 0) {
        cout << "Error: n must be divisible by block_size!" << endl;
        return {};
    }
    int k = n / block_size;
    vector<vector<double>> result(n, vector<double>(n, 0.0));
    int row_offset = 0, col_offset = 0;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(min_val, max_val);
    for (int i = 0; i < k; i++) {
        vector<vector<double>> block(block_size, vector<double>(block_size, 0.0));
        for (int j = 0; j < block_size; j++) {
            for (int l = 0; l < block_size; l++) {
                block[j][l] = dis(gen);
            }
        }
        for (int j = 0; j < block_size; j++) {
            for (int l = 0; l < block_size; l++) {
                result[row_offset + j][col_offset + l] = block[j][l];
            }
        }
        row_offset += block_size;
        col_offset += block_size;
    }
    return result;
}



void initialize(vector<vector<double>>& U, vector<vector<double>>& V, int k) {
    int n = U.size(), m = V.size();
    for (int i = 0; i < n; i++) {
        for (int r = 0; r < k; r++) {
            U[i][r] = (double) rand() / RAND_MAX;
        }
    }
    for (int j = 0; j < m; j++) {
        for (int r = 0; r < k; r++) {
            V[j][r] = (double) rand() / RAND_MAX;
        }
    }
}
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    int n = matrix.size();
    int m = matrix[0].size();
    std::vector<std::vector<double>> transposed(m, std::vector<double>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

double calculate_factorization_rmse(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& U, const std::vector<std::vector<double>>& V) {
    int m = A.size();
    int n = A[0].size();

    double sum = 0.0;
    int count = 0;

    int r = U[0].size(); // number of columns in U

    // check dimensions of U and V
    if (U.size() != m || V.size() != r || V[0].size() != n) {
        std::cout << "Error: invalid dimensions of U and/or V!" << std::endl;
        std::cout << U.size() << r<<V.size()<<V[0].size()<<m<<n <<std::endl;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] != 0) {
                double A_hat_ij = 0.0;
                for (int k = 0; k < U[i].size(); k++) {
                    A_hat_ij += U[i][k] * V[k][j];
                }
                sum += pow(A[i][j] - A_hat_ij, 2);
                count++;
            }
        }
    }

    if (count == 0) {
        return 0;
    } else {
        return sqrt(sum / count);
    }
}


vector<vector<double>> matrix_product(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    vector<vector<double>> C(m, vector<double>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

vector<vector<double>> get_block(const vector<vector<double>>& D, int block_size, int block_num) {
    vector<vector<double>> block(block_size, vector<double>(block_size, 0.0));
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            block[i][j] = D[block_num * block_size + i][block_num * block_size + j];
        }
    }
    return block;
}

double get_error(int i, int j, const vector<vector<double>>& U, const vector<vector<double>>& V) {
    int k = U[0].size();
    double error = 0;
    for (int r = 0; r < k; r++) {
        error += U[i][r] * V[j][r];
    }
    return error;
}

class BlockScheduler {
public:
    BlockScheduler(int num_blocks) : free_blocks(num_blocks), num_updates(num_blocks, 0), in_use(num_blocks, false) {
        iota(free_blocks.begin(), free_blocks.end(), 0); // Initialize free_blocks with 0, 1, 2, ..., num_blocks - 1
    }

    int get_block() {
        lock_guard<mutex> lock(scheduler_mutex);

        // If no free blocks, return -1
        if (all_of(in_use.begin(), in_use.end(), [](bool v) { return v; })) {
            //cout << "No free blocks available\n";
            return -1;
        }

        // Find the smallest number of updates
        int min_updates = *min_element(num_updates.begin(), num_updates.end());

        // Log the minimum number of updates
        //cout << "Minimum number of updates: " << min_updates << "\n";

        // Select all blocks with the smallest number of updates
        vector<int> candidates;
        for (int b : free_blocks) {
            if (!in_use[b] && num_updates[b] == min_updates) {
                candidates.push_back(b);
            }
        }

        // If no candidates, there is a problem
        if (candidates.empty()) {
            //cout << "No blocks with minimum updates found\n";
            return -1;
        }

        // Randomly select a block from the candidates
        uniform_int_distribution<int> dist(0, candidates.size() - 1);
        int selected_block = candidates[dist(gen)];

        // Mark the block as in use
        in_use[selected_block] = true;

        return selected_block;
    }


    // This function is equivalent to the 'put job' procedure in Algorithm 5
    void put_block(int block) {
        lock_guard<mutex> lock(scheduler_mutex);
        // Increase the block's update times by one
        num_updates[block]++;
        // Mark the block as not in use
        in_use[block] = false;
    }

private:
    mutex scheduler_mutex;
    vector<int> free_blocks;
    vector<int> num_updates;
    vector<bool> in_use;
    random_device rd;
    mt19937 gen{rd()};
};


void dsgd_threaded(int block_size, int num_threads, int num_iterations, double step_size, double lambda, const vector<vector<double>>& A, vector<vector<double>>& U, vector<vector<double>>& V) {
    int n = A.size(), m = A[0].size(), k = U[0].size();
    std::ofstream output_file("fsgd_results.csv");
    auto start_time = std::chrono::high_resolution_clock::now();
    output_file << "Iteration,RMSE,Time" << std::endl;
    vector<thread> threads(num_threads);
    int num_blocks = n / block_size;

    // Create a scheduler
    BlockScheduler scheduler(num_blocks);

    for (int t = 0; t < num_threads; t++) {
        threads[t] = thread([&](int thread_id) {
            for (int iter = 0; iter < num_iterations; iter++) {
                // Get a block from the scheduler
                int block_num = scheduler.get_block();
                if (block_num == -1) continue;  // No free blocks

                int start_row = block_num * block_size;
                int end_row = min(start_row + block_size, n);

                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < m; j++) {
                        if (A[i][j] != 0) {
                            double error = A[i][j] - get_error(i, j, U, V);
                            for (int r = 0; r < k; r++) {
                                U[i][r] += step_size * (error * V[j][r] - lambda * U[i][r]);
                                V[j][r] += step_size * (error * U[i][r] - lambda * V[j][r]);
                            }
                        }
                    }
                }

                // Return the block to the scheduler
                scheduler.put_block(block_num);

                if (iter % 50 == 0) {
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                    double rmse = calculate_factorization_rmse(A, U, transpose(V));
                    output_file << iter << "," << rmse << "," << elapsed_time << std::endl;
                }
            }
        }, t);
    }

    for (int t = 0; t < num_threads; t++) {
        threads[t].join();
    }
}


void print_matrix(vector<vector<double>>& matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


int main() {
    int num_iterations = 1000;
    double step_size = 0.01, lambda = 0.01;
    vector<thread> threads;
    int n = 500, m = 500, k = 50;
    double sparsity = 0.5;

    int block_size = 10;
    int num_threads = 10;
    int min_val = 0;
    int max_val = 1;
    vector<vector<double>> A = block_diagonal_matrix(n, block_size, min_val, max_val);
    vector<vector<double>> U(n, vector<double>(k)), V(m, vector<double>(k));

    initialize(U, V, k);


    double rmse = calculate_factorization_rmse(A, U, transpose(V));
    dsgd_threaded(block_size, num_threads, num_iterations, step_size, lambda, A, U, V);
    vector<vector<double>> product = matrix_product(U, transpose(V));
    // Calculate the RMSE between A and the factorization of A by U and V
    std::cout << "RMSE = " << rmse << "\n";
    double rmse_final = calculate_factorization_rmse(A, U, transpose(V));

    // Print the RMSE to the console
    std::cout << "RMSE FINAL = " << rmse_final << "\n";



    return 0;
}