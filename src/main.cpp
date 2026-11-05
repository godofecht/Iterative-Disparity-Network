#include "DisparityNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try
    {
        std::string left_path = "image1.csv";
        std::string right_path = "image2.csv";
        std::string truth_path = "disparity.csv";
        std::string output_dir = "results";
        unsigned epochs = 2000, seed = 42;
        std::size_t stride = 7;
        double short_half_life = 32.0, long_half_life = 3200.0;
        for (int i = 1; i < argc; ++i)
        {
            const std::string key(argv[i]);
            if (key == "--help")
            {
                std::cout << "Usage: DisparityNetwork [--left FILE] [--right FILE] "
                             "[--reference FILE] [--output-dir DIR] [--epochs N] "
                             "[--seed N] [--stride N] [--short-half-life H] "
                             "[--long-half-life H]\n";
                return 0;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + key);
            const std::string value(argv[++i]);
            if (key == "--left") left_path = value;
            else if (key == "--right") right_path = value;
            else if (key == "--reference") truth_path = value;
            else if (key == "--output-dir") output_dir = value;
            else if (key == "--epochs") epochs = static_cast<unsigned>(std::stoul(value));
            else if (key == "--seed") seed = static_cast<unsigned>(std::stoul(value));
            else if (key == "--stride") stride = static_cast<std::size_t>(std::stoul(value));
            else if (key == "--short-half-life") short_half_life = std::stod(value);
            else if (key == "--long-half-life") long_half_life = std::stod(value);
            else throw std::invalid_argument("Unknown option: " + key);
        }
        const auto left = disparity::read_numbers(left_path);
        const auto right = disparity::read_numbers(right_path);
        const auto samples = disparity::make_patches(left, right, stride);
        const disparity::Network model(samples, short_half_life, long_half_life);
        auto weights = disparity::Network::initial_weights(seed);
        std::filesystem::create_directories(output_dir);

        // Save actual intermediate network outputs. These snapshots, not interpolations,
        // drive the site's time slider and can be reproduced from the training seed.
        std::ofstream checkpoints(output_dir + "/checkpoints.csv");
        if (!checkpoints)
            throw std::runtime_error("Cannot write disparity checkpoints");
        checkpoints << std::setprecision(17);
        auto save_checkpoint = [&](unsigned epoch, const disparity::Weights& current_weights)
        {
            const auto values = model.evaluate(current_weights, false).predictions;
            checkpoints << epoch;
            for (const double value : values)
                checkpoints << ',' << value;
            checkpoints << '\n';
            if (!checkpoints)
                throw std::runtime_error("Cannot write disparity checkpoint row");
        };
        const auto history = model.train(weights, epochs, save_checkpoint);
        if (history.back().iteration % 50 != 0 && history.back().iteration != epochs)
            save_checkpoint(history.back().iteration, weights);
        checkpoints.close();
        const auto final = model.evaluate(weights);
        disparity::write_numbers(output_dir + "/weights.csv", "weight",
                                 std::vector<double>(weights.begin(), weights.end()));
        disparity::write_numbers(output_dir + "/predictions.csv", "prediction", final.predictions);
        std::ofstream metrics(output_dir + "/training_metrics.csv");
        if (!metrics)
            throw std::runtime_error("Cannot write training metrics");
        metrics << "epoch,merit,short_variance,long_variance,gradient_norm,step\n"
                << std::setprecision(17);
        for (const auto& epoch : history)
            metrics << epoch.iteration << ',' << epoch.merit << ','
                    << epoch.short_variance << ',' << epoch.long_variance << ','
                    << epoch.gradient_norm << ',' << epoch.step << '\n';
        metrics.close();
        std::cout << std::setprecision(10) << "samples=" << samples.size()
                  << " epochs_completed=" << history.back().iteration
                  << " initial_F=" << history.front().merit
                  << " final_F=" << final.merit;
        if (std::filesystem::exists(truth_path))
        {
            const auto reference = disparity::read_numbers(truth_path);
            const double r = disparity::correlation(final.predictions, reference);
            std::cout << " disparity_r=" << r << " disparity_abs_r=" << std::abs(r);
        }
        std::cout << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "DisparityNetwork: " << error.what() << '\n';
        return 1;
    }
}
