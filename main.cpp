#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <tuple>
#include <ctime>
#include <fstream>
#include <atomic>


const int num_factors = 100;
const int num_iterations = 1000000;
const double learning_rate = 0.001;
const double lambda_p = 0.1;
const double lambda_q = 0.1;
const int num_threads = 2;

double dot_product(const std::vector<double> &a, const std::vector<double> &b) {
    double result = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}


void hogwild(const std::vector<std::tuple<int, int, double>> &ratings,
             std::vector<std::vector<double>> &P,
             std::vector<std::vector<double>> &Q) {
    std::default_random_engine generator(std::time(nullptr));
    std::uniform_int_distribution<int> distribution(0, ratings.size() - 1);


    for (int i = 0; i < num_iterations; ++i) {
        // randomly select an index
        int idx = distribution(generator);
        int user = std::get<0>(ratings[idx]); // Get the user corresponding to the randomly selected index
        int item = std::get<1>(ratings[idx]); // Get the item corresponding to the randomly selected index
        double rating = std::get<2>(ratings[idx]); // Get the rating corresponding to the randomly selected index

        double prediction = dot_product(P[user], Q[item]);
        double error = rating - prediction;

        for (int k = 0; k < num_factors; ++k) {
            double p_u = P[user][k];
            double q_v = Q[item][k];

            P[user][k] += learning_rate * (error * q_v - lambda_p * p_u);
            Q[item][k] += learning_rate * (error * p_u - lambda_q * q_v);
        }

    }
    //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "Thread complete: " << std::endl;
}


std::vector<std::tuple<int, int, double>> generate_synthetic_data(int num_users, int num_items, double sparsity) {
    int num_non_zero_entries = static_cast<int>((1.0 - sparsity) * num_users * num_items);

    std::vector<std::tuple<int, int, double>> synthetic_data;
    synthetic_data.reserve(num_non_zero_entries);

    std::default_random_engine generator(std::time(nullptr));
    std::uniform_int_distribution<int> user_distribution(0, num_users - 1);
    std::uniform_int_distribution<int> item_distribution(0, num_items - 1);
    std::uniform_real_distribution<double> rating_distribution(1.0, 5.0);

    for (int i = 0; i < num_non_zero_entries; ++i) {
        int user = user_distribution(generator);
        int item = item_distribution(generator);
        double rating = rating_distribution(generator);

        synthetic_data.emplace_back(user, item, rating);
    }

    return synthetic_data;
}


std::atomic<bool> stop_monitoring(false);
std::mutex rmse_data_mutex;

std::atomic<bool> monitoring_started(false);

void monitor_progress(const std::vector<std::tuple<int, int, double>> &ratings,
                      const std::vector<std::vector<double>> &P,
                      const std::vector<std::vector<double>> &Q,
                      std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
                      std::vector<std::pair<long long, double>> &rmse_data) {
    int monitoring_interval_ms = 0.01; // Choose the interval for monitoring progress (in milliseconds)

    while (!stop_monitoring.load()) {
        monitoring_started.store(true); // Indicate that monitoring has started
        //std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms));

        double rmse = 0;

        for (const auto &entry : ratings) {
            int user = std::get<0>(entry);
            int item = std::get<1>(entry);
            double rating = std::get<2>(entry);

            double prediction = dot_product(P[user], Q[item]);
            double error = rating - prediction;

            rmse += error * error;
        }

        rmse = sqrt(rmse / ratings.size());

        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

        std::cout << "Time: " << time_since_start << " ms, RMSE: " << rmse << std::endl;

        // Acquire lock to avoid data race
        std::lock_guard<std::mutex> lock(rmse_data_mutex);
        rmse_data.push_back(std::make_pair(time_since_start, rmse));
        std::cout << "stop_monitoring: " << stop_monitoring << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms));

    }
}





int main() {
    int num_users = 2000;
    int num_items = 2000;
    double sparsity = 0.7;

    std::vector<std::tuple<int, int, double>> ratings = generate_synthetic_data(num_users, num_items, sparsity);

    std::vector<std::vector<double>> P(num_users, std::vector<double>(num_factors, 0.1));
    std::vector<std::vector<double>> Q(num_items, std::vector<double>(num_factors, 0.1));

    std::vector<std::thread> threads;

    // Record the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<long long, double>> rmse_data;

    std::thread monitoring_thread(monitor_progress, std::ref(ratings), std::ref(P), std::ref(Q), start_time, std::ref(rmse_data));

    // Wait for monitoring to start
    while (!monitoring_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(hogwild, std::ref(ratings), std::ref(P), std::ref(Q));
    }


    for (auto &thread: threads) {
        thread.join();
    }

    stop_monitoring.store(true);
    monitoring_thread.join();

    // Record the end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate and print the elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "HogWild! algorithm took " << duration << " milliseconds to execute." << std::endl;


    std::ofstream rmse_file("rmse_data.csv");
    for (const auto &entry: rmse_data) {
        rmse_file << entry.first << "," << entry.second << std::endl;
    }
    rmse_file.close();

    return 0;
}
